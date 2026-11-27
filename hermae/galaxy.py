"""Two-head galaxy segmentation and centre prediction."""

from hashlib import file_digest

import numpy as np
from scipy import ndimage
from skimage.segmentation import watershed
import torch
from torch import nn

from .detection import Patch, tile_starts


def block(inputs, outputs):
    return nn.Sequential(nn.Conv2d(inputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU(),
                         nn.Conv2d(outputs, outputs, 3, padding=1), nn.GroupNorm(4, outputs), nn.SiLU())


class GalaxyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.low = block(2, 16)
        self.medium = block(16, 32)
        self.high = block(32, 64)
        self.decode_medium = block(64 + 32, 32)
        self.decode_low = block(32 + 16, 16)
        self.footprint = nn.Conv2d(16, 1, 1)
        self.centre = nn.Conv2d(16, 1, 1)

    def forward(self, image):
        def join(deep, skip):
            deep = nn.functional.interpolate(deep, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            return torch.cat((deep, skip), dim=1)

        low = self.low(image)
        medium = self.medium(nn.functional.max_pool2d(low, 2))
        high = self.high(nn.functional.max_pool2d(medium, 2))
        decoded = self.decode_low(join(self.decode_medium(join(high, medium)), low))
        return torch.cat((self.footprint(decoded), self.centre(decoded)), dim=1)


def pixels(valid, signal, variance):
    normalized = np.arcsinh(np.divide(signal, np.sqrt(variance), out=np.zeros_like(signal),
                                      where=valid & (variance > 0)))
    return np.stack((np.clip(normalized, -5, 9), valid), dtype=np.float32)


def pad_tensor(array, device):
    height, width = array.shape[-2:]
    value = torch.from_numpy(np.ascontiguousarray(array)).to(device)
    return nn.functional.pad(value, (0, (-width) % 4, 0, (-height) % 4))


def instances(footprint, centre, valid, minimum, footprint_threshold, centre_threshold):
    region = valid & (footprint >= footprint_threshold)
    peaks = (region & (centre >= centre_threshold)
             & (centre == ndimage.maximum_filter(centre, size=5)))
    peak_labels, count = ndimage.label(peaks, structure=np.ones((3, 3)))
    markers = np.zeros(region.shape, dtype=np.int32)
    for i, bounds in enumerate(ndimage.find_objects(peak_labels), 1):
        patch = np.where(peak_labels[bounds] == i, centre[bounds], -np.inf)
        y, x = np.unravel_index(patch.argmax(), patch.shape)
        markers[y + bounds[0].start, x + bounds[1].start] = i
    if count == 0:
        return []
    labels = watershed(-footprint, markers, mask=region)
    result = []
    for marker, bounds in enumerate(ndimage.find_objects(labels), 1):
        if bounds is None:
            continue
        y, x = np.nonzero(labels[bounds] == marker)
        if len(x) < minimum:
            continue
        yy, xx = np.nonzero(markers[bounds] == marker)
        result.append((y + bounds[0].start, x + bounds[1].start,
                       (int(xx[0]) + bounds[1].start, int(yy[0]) + bounds[0].start)))
    return result


class GalaxyDetector:
    def __init__(self, model, device, footprint_threshold, centre_threshold, checkpoint_sha256=None):
        self.model = model.to(device).eval()
        self.device = device
        self.footprint_threshold = footprint_threshold
        self.centre_threshold = centre_threshold
        self.checkpoint_sha256 = checkpoint_sha256

    @classmethod
    def load(cls, path, device="cpu"):
        with path.open("rb") as stream:
            content_hash = file_digest(stream, "sha256").hexdigest()
            stream.seek(0)
            state = torch.load(stream, map_location="cpu", weights_only=True)
        if state.get("kind") != "hermae-galaxy-v1":
            raise ValueError("Expected a Hermae galaxy detector checkpoint.")
        model = GalaxyNet()
        model.load_state_dict(state["model"])
        return cls(model, device, state["footprint_threshold"], state["centre_threshold"], content_hash)

    @torch.inference_mode()
    def predict(self, image):
        height, width = image.shape[-2:]
        value = pad_tensor(image[None], self.device)
        return self.model(value).sigmoid()[0, :, :height, :width].cpu().numpy()

    def patches(self, frame, config, signal, variance):
        height, width = frame.science.shape
        image = pixels(frame.valid, signal, variance)
        patches = []
        for tile_id, (y, x) in enumerate(((y, x) for y in tile_starts(height, config.tile, config.overlap)
                                         for x in tile_starts(width, config.tile, config.overlap)), 1):
            region = np.s_[y:y + config.tile, x:x + config.tile]
            valid = frame.valid[region]
            if not valid.any():
                continue
            footprint, centre = self.predict(image[:, region[0], region[1]])
            for yy, xx, (cx, cy) in instances(footprint, centre, valid, config.min_pixels,
                                               self.footprint_threshold, self.centre_threshold):
                patches.append(Patch(tile_id, (yy + y) * width + xx + x, (cx + x, cy + y)))
        return patches
