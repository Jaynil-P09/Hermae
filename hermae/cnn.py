"""Train matched catalogue, pixel and fused host rankers on the frozen split."""

import argparse
from hashlib import file_digest
from importlib.metadata import version
import json
from pathlib import Path

from astropy.coordinates import SkyCoord
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn
from torch.nn import functional as F

from .benchmark import FEATURES, RAW_FEATURES, catalogue_features, paired_interval
from .image_benchmark import inputs
from .imaging import SN_MASK_ARCSEC, read_image
from .survey import PIX_SCALE

STAMP = 21
EPOCHS = 14
BATCH_FIELDS = 8
LEARNING_RATE = 0.003
WEIGHT_DECAY = 0.01
SEED = 42
MODES = ("catalogue_control", "cnn_only", "cnn_fusion")
KEYS = ["event_id", "source_id"]


def pixel_stamps(image, valid, centres):
    """Extract zero-padded flux and coverage channels at WCS-projected positions."""
    offset = np.arange(STAMP) - STAMP // 2
    x = np.rint(centres[..., 0]).astype(int)[..., None, None] + offset[None, :]
    y = np.rint(centres[..., 1]).astype(int)[..., None, None] + offset[:, None]
    inside = (x >= 0) & (y >= 0) & (x < image.shape[1]) & (y < image.shape[0])
    x = np.clip(x, 0, image.shape[1] - 1)
    y = np.clip(y, 0, image.shape[0] - 1)
    mask = inside & valid[y, x]
    return np.stack((np.where(mask, image[y, x], 0.), mask), axis=-3).astype("float32")


def stamp_centres(centres, sn):
    return np.stack((centres, np.broadcast_to(sn, centres.shape), (centres + sn) / 2), axis=1)


def load_stamps(images, cache, events, candidates):
    audit = pd.read_csv(images / "image_inputs.csv").set_index("event_id")
    if (not audit.index.is_unique or set(audit.index) != set(events.event_id)
            or not audit.status.isin(["ok", "no_r_band_weight"]).all()):
        raise ValueError("Image audit must cover exactly the verified training and validation fields.")
    if not np.allclose(audit.pixel_scale_arcsec.to_numpy(), PIX_SCALE, rtol=0, atol=1e-6):
        raise ValueError("CNN v1 requires 0.262″ per pixel.")
    pixels = np.zeros((len(candidates), 3, 2, STAMP, STAMP), dtype="float32")
    for event in events.itertuples():
        record = audit.loc[event.event_id]
        if record.split != event.split:
            raise ValueError(f"Image split mismatch for {event.event_id}.")
        path = cache / "image_ablation" / record.cache_file
        with path.open("rb") as stream:
            if file_digest(stream, "sha256").hexdigest() != record.sha256:
                raise ValueError(f"FITS bytes differ from the audit for {event.event_id}.")
        if record.status != "ok":
            continue
        pool = candidates.loc[candidates.event_id.eq(event.event_id)]
        if pool.empty:
            continue
        image, valid, wcs = read_image(path, event, PIX_SCALE)
        sn = np.array(wcs.world_to_pixel(SkyCoord(event.ra, event.dec, unit="deg")))
        yy, xx = np.ogrid[:image.shape[0], :image.shape[1]]
        valid &= np.hypot(xx - sn[0], yy - sn[1]) * PIX_SCALE > SN_MASK_ARCSEC
        if not valid.any():
            continue
        sky = np.median(image[valid])
        noise = 1.4826 * np.median(np.abs(image[valid] - sky))
        if not np.isfinite(noise) or noise <= 0:
            continue
        scaled = np.arcsinh((image - sky) / noise) / 4.
        centres = np.column_stack(wcs.world_to_pixel(SkyCoord(pool.ra.to_numpy(),
                                                             pool.dec.to_numpy(), unit="deg")))
        locations = stamp_centres(centres, sn)
        pixels[pool.index] = pixel_stamps(scaled, valid, locations)
    return torch.from_numpy(pixels)


class HostRanker(nn.Module):
    def __init__(self, width, mode):
        super().__init__()
        self.tabular = nn.Linear(width, 1, bias=False) if mode != "cnn_only" else None
        self.encoder = None
        if mode != "catalogue_control":
            self.encoder = nn.Sequential(
                nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
                nn.Conv2d(8, 12, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
            )
            self.image_head = nn.Sequential(nn.Linear(144, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, table, pixels, has_pixels):
        catalogue = self.tabular(table).squeeze(1) if self.tabular is not None else table.new_zeros(len(table))
        image = table.new_zeros(len(table))
        if self.encoder is not None:
            embed = self.encoder(pixels.flatten(0, 1)).reshape(len(table), -1)
            image = self.image_head(embed).squeeze(1) * has_pixels
        return catalogue, image


def fit(mode, table, pixels, coverage, fields):
    torch.manual_seed(SEED)
    order = np.random.default_rng(SEED)
    model = HostRanker(table.shape[1], mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    losses = []
    for epoch in range(EPOCHS):
        permutation = order.permutation(len(fields))
        epoch_loss = 0.
        for start in range(0, len(fields), BATCH_FIELDS):
            batch = [fields[i] for i in permutation[start:start + BATCH_FIELDS]]
            sizes = [len(indices) for indices, _ in batch]
            indices = np.concatenate([indices for indices, _ in batch])
            cat, img = model(table[indices], pixels[indices] if model.encoder is not None else None,
                             coverage[indices])
            logits = nn.utils.rnn.pad_sequence((cat + img).split(sizes), batch_first=True,
                                               padding_value=-torch.inf)
            loss = F.cross_entropy(logits, torch.tensor([target for _, target in batch]))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
        losses.append(epoch_loss / len(fields))
        if epoch in {0, EPOCHS - 1}:
            print(f"{mode} epoch {epoch + 1}/{EPOCHS} train loss {losses[-1]:.4f}", flush=True)
    return model.eval(), losses


def run_benchmark(run, images, cache, manifest, output):
    events, _ = inputs(run, manifest)
    table_rows = pd.read_csv(run / "candidates.csv", usecols=[*RAW_FEATURES, "in_baseline_cone"])
    table_rows = table_rows.loc[table_rows.event_id.isin(events.event_id)]
    features = pd.concat([catalogue_features(field) for _, field in table_rows.groupby("event_id")])
    candidates = table_rows.loc[table_rows.in_baseline_cone].copy()
    columns = FEATURES["catalogue"]
    features = features.loc[candidates.index, columns].reset_index(drop=True)
    candidates = candidates.reset_index(drop=True)
    train_ids = set(events.loc[events.split.eq("train"), "event_id"])
    val_ids = set(events.loc[events.split.eq("validation"), "event_id"])
    reference = pd.read_csv(run / "evaluation.csv", usecols=["event_id", "reference_source_id"])
    retrieved_col = f"reference_retrieved_{manifest['search_radii_arcsec'][0]}as"
    eval_rows = pd.read_csv(run / "evaluation.csv", usecols=["event_id", retrieved_col])
    references = reference.merge(eval_rows, on="event_id", validate="one_to_one")
    references = references.loc[references.event_id.isin(train_ids) & references[retrieved_col].eq(True)]
    hosts = references.set_index("event_id").reference_source_id
    fields = []
    for event_id, rows in candidates.loc[candidates.event_id.isin(hosts.index)].groupby("event_id"):
        targets = np.flatnonzero(rows.source_id.eq(hosts.loc[event_id]))
        if len(targets) != 1:
            raise ValueError(f"Expected one training counterpart in {event_id}.")
        if len(rows) > 1:
            fields.append((rows.index.to_numpy(), int(targets[0])))
    if not fields:
        raise ValueError("No informative training fields.")
    prep = make_pipeline(SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True),
                         StandardScaler())
    prep.fit(features.iloc[np.concatenate([indices for indices, _ in fields])])
    table = torch.tensor(prep.transform(features), dtype=torch.float32)
    pixels = load_stamps(images, cache, events, candidates)
    coverage = pixels[:, 0, 1].mean(dim=(-2, -1)).ge(0.5)
    val = candidates.loc[candidates.event_id.isin(val_ids), [*KEYS, "separation_arcsec"]].copy()
    val["cnn_has_pixels"] = coverage[val.index.to_numpy(copy=True)].numpy()
    summary = pd.read_csv(images / "image_validation.csv").set_index("event_id")
    if not summary.index.is_unique or set(summary.index) != val_ids:
        raise ValueError("Aperture benchmark validation events differ from the frozen split.")
    summary = summary.sort_index()
    prior = pd.read_csv(images / "image_rankings.csv").set_index(KEYS)
    expected = pd.MultiIndex.from_frame(val[KEYS])
    if not prior.index.is_unique or len(prior) != len(val) or not prior.index.difference(expected).empty:
        raise ValueError("Aperture benchmark rankings differ from the frozen validation pool.")

    imputer, scaler = prep.steps[0][1], prep.steps[1][1]
    checkpoint = dict(
        format_version=1, columns=columns, stamp_pixels=STAMP, pixel_scale_arcsec=PIX_SCALE,
        medians=imputer.statistics_.tolist(), indicators=imputer.indicator_.features_.tolist(),
        means=scaler.mean_.tolist(), scales=scaler.scale_.tolist(), models={},
    )
    logs = []
    for mode in MODES:
        model, losses = fit(mode, table, pixels, coverage, fields)
        logs.extend(dict(model=mode, epoch=i, training_loss=loss) for i, loss in enumerate(losses, 1))
        checkpoint["models"][mode] = model.state_dict()
        indices = val.index.to_numpy(copy=True)
        chunks = []
        with torch.inference_mode():
            for chunk in np.array_split(indices, max(1, int(np.ceil(len(indices) / 256)))):
                cat, img = model(table[chunk], pixels[chunk] if model.encoder is not None else None,
                                 coverage[chunk])
                chunks.append(torch.stack((cat + img, img), dim=1).numpy())
        scores = np.concatenate(chunks)
        val[f"{mode}_score"] = scores[:, 0]
        if mode == "cnn_fusion":
            val["cnn_fusion_image_term"] = scores[:, 1]
        val = val.sort_values(["event_id", f"{mode}_score", "separation_arcsec", "source_id"],
                              ascending=[True, False, True, True])
        rank = f"{mode}_rank"
        val[rank] = val.groupby("event_id").cumcount() + 1
        lookup = val.set_index(KEYS)[rank]
        keys = pd.MultiIndex.from_arrays([summary.index, summary.reference_source_id])
        summary[f"{mode}_reference_rank"] = lookup.reindex(keys).to_numpy()
    output.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output / "rankers.pt")
    pd.DataFrame(logs).to_csv(output / "cnn_training.csv", index=False)
    val.to_csv(output / "cnn_rankings.csv", index=False)
    summary.to_csv(output / "cnn_validation.csv")
    return summary, val, len(fields), retrieved_col


def report(summary, rankings, train_count, retrieved):
    recovered = summary.loc[summary[retrieved].eq(True)]
    names = {
        "catalogue": "Catalogue logistic regression",
        "image": "Catalogue + measured apertures",
        "catalogue_control": "Matched catalogue control",
        "cnn_only": "CNN only",
        "cnn_fusion": "Catalogue + CNN",
    }
    lines = [
        "# Pixel CNN and catalogue fusion", "",
        f"The three listwise rankers used {train_count} informative training fields and scored the same "
        f"{len(summary)} validation fields and {len(rankings):,} candidates. The reserved test set remained untouched.",
        f"The CNN received usable pixels for {rankings.cnn_has_pixels.sum():,} validation candidates. "
        "For missing stamps the CNN score term is zero. CNN-only ties use angular separation.",
        "",
        f"## Agreement among {len(recovered)} uniquely recovered published counterparts", "",
    ]
    for key, name in names.items():
        ranks = recovered[f"{key}_reference_rank"]
        lines.append(f"- **{name}** ranked {ranks.eq(1).sum()}/{len(recovered)} first "
                     f"and {ranks.le(3).sum()}/{len(recovered)} in the top three.")
    if not recovered.empty:
        delta = paired_interval(recovered, "cnn_fusion", "catalogue_control", SEED)
        lines += ["", f"The paired first-rank change from adding the CNN to the matched control was {delta}."]
    lines += ["", f"The CPU experiment used PyTorch {version('torch')}, seed {SEED}, {EPOCHS} epochs, "
              f"AdamW with learning rate {LEARNING_RATE}, weight decay {WEIGHT_DECAY}, and "
              f"{BATCH_FIELDS} fields per update.",
              "It used r-band stamps centred on the candidate, SN and midpoint, each with a coverage mask. "
              "The three stamps share two convolutional layers. The fused score adds a learned image term "
              "to the catalogue score. Redshift was omitted because the initial audit found insufficient "
              "independent overlap. Published ZTF counterparts remain catalogue references.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("results/benchmark"))
    parser.add_argument("--images", type=Path, default=Path("results/benchmark/images"))
    parser.add_argument("--split", type=Path, default=Path("benchmark/split-v1.json"))
    parser.add_argument("--cache", type=Path, default=Path(".cache"))
    parser.add_argument("--output", type=Path, default=Path("results/benchmark/cnn"))
    options = parser.parse_args()
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    manifest = json.loads(options.split.read_text(encoding="utf-8"))
    results = run_benchmark(options.run, options.images, options.cache, manifest, options.output)
    (options.output / "cnn_report.md").write_text(report(*results), encoding="utf-8")
    print(f"CNN validation written to {options.output}.")


if __name__ == "__main__":
    main()
