"""Candidate cutouts and an image-first field ranking model."""

import numpy as np
from scipy import ndimage
import torch
from torch import nn

from .development import digest


class HostRanker(nn.Module):
    def __init__(self, use_catalogue=False):
        super().__init__()
        self.use_catalogue = use_catalogue
        self.encoder = nn.Sequential(nn.Conv2d(4, 16, 5, padding=2), nn.SiLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(16, 24, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
                                     nn.Conv2d(24, 32, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(3),
                                     nn.Flatten())
        self.score = nn.Sequential(nn.Linear(32 * 9 * 2 + 8 + (8 if use_catalogue else 0), 48),
                                   nn.SiLU(), nn.Linear(48, 1))

    def forward(self, local, wide, geometry, catalogue):
        features = [self.encoder(local), self.encoder(wide), geometry]
        if self.use_catalogue:
            features.append(catalogue)
        return self.score(torch.cat(features, dim=1)).squeeze(1)


def cutout(image, labels, source_id, cx, cy, sn_x, sn_y, step):
    axis = np.arange(32) - 15.5
    yy, xx = np.meshgrid(cy + axis * step, cx + axis * step, indexing="ij")
    coords = np.stack((yy, xx))
    valid = ndimage.map_coordinates(image[1], coords, order=0, mode="constant", cval=0)
    values = ndimage.map_coordinates(image[0], coords, order=1, mode="constant", cval=0) * valid
    sampled = ndimage.map_coordinates(labels, coords, order=0, mode="constant", cval=0)
    footprint = (sampled == source_id) & (source_id > 0)
    marker = np.exp(-((xx - sn_x)**2 + (yy - sn_y)**2) / (2 * step**2))
    return np.stack((values, footprint, marker * valid, valid), dtype=np.float32)


def candidate_inputs(image, labels, ranked, detections, sn_x, sn_y, catalogue):
    result = [], [], [], []
    for row in ranked:
        source_id = row["source_id"]
        d = detections.get(source_id, {})
        x, y = row["x_px"], row["y_px"]
        offset = np.hypot(x - sn_x, y - sn_y)
        result[0].append(cutout(image, labels, source_id, x, y, sn_x, sn_y, 1))
        result[1].append(cutout(image, labels, source_id, (x + sn_x) / 2, (y + sn_y) / 2,
                                sn_x, sn_y, max(2., offset / 23.)))
        signal = d.get("flux", 0.) / max(d.get("flux_noise", 0.), 1e-12) if d else 0.
        result[2].append([np.log1p(row["separation_arcsec"]), np.log1p(row["size_scaled_offset"] or 0.),
                          bool(row["shape_fallback"]), np.log1p(d.get("major_sigma_px", 0.)),
                          np.log1p(d.get("minor_sigma_px", 0.)), np.arcsinh(signal) / 4.,
                          np.log1p(d.get("footprint_pixels", 0.)) / 4., bool(d)])
        result[3].append(catalogue_features(catalogue.get(source_id)))
    return tuple(torch.as_tensor(np.asarray(values, dtype=np.float32)) for values in result)


def catalogue_features(row):
    if row is None:
        return [0.] * 8
    g, r, z = [float(row[key] or 0.) for key in ("flux_g", "flux_r", "flux_z")]
    return [1., np.arcsinh(r), np.arcsinh(g) - np.arcsinh(r), np.arcsinh(r) - np.arcsinh(z),
            bool(float(row["flux_ivar_g"] or 0.)), bool(float(row["flux_ivar_z"] or 0.)),
            np.log1p(float(row["shape_r"] or 0.)), np.log1p(float(row["dlr"] or 0.))]


def load_ranker(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("kind") != "hermae-host-v1":
        raise ValueError("Expected a Hermae host ranker checkpoint.")
    model = HostRanker(state["use_catalogue"])
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state, digest(path)


@torch.inference_mode()
def score(model, tensors, device):
    return model(*(tensor.to(device) for tensor in tensors)).cpu().numpy()


def rank_image_hosts(path, device, frame, labels, signal, variance, ranked, detections, x, y, backend,
                     pipeline=None):
    from .galaxy import pixels

    model, state, identity = load_ranker(path, device) if pipeline is None else pipeline
    if state["use_catalogue"]:
        raise ValueError("The FITS-only command requires an image-only ranker checkpoint.")
    expected = state["detector_sha256"]
    actual = backend.checkpoint_sha256 if backend else None
    if expected != actual:
        raise ValueError("The ranker and detector checkpoints were trained for different candidate pools.")
    gap = None
    raw = []
    if ranked:
        values = candidate_inputs(pixels(frame.valid, signal, variance), labels, ranked,
                                  {r["source_id"]: r for r in detections}, x, y, {})
        raw = score(model, values, device).tolist()
        if not np.isfinite(raw).all():
            raise ValueError("The host ranker produced nonfinite scores.")
        for row, value in zip(ranked, raw):
            row["ranker_score"] = value
        ranked.sort(key=lambda row: (-row["ranker_score"], row["rank"]))
        for i, row in enumerate(ranked, 1):
            row["rank"] = i
        if len(ranked) > 1:
            gap = ranked[0]["ranker_score"] - ranked[1]["ranker_score"]
    threshold = state["review_gap"]
    review = bool(ranked and (threshold is None or (gap is not None and gap < threshold)))
    return dict(sha256=identity, score_gap=gap, review_gap=threshold, review_required=review)
