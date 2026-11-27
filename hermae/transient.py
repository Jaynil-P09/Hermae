"""CNN vetting for residuals proposed by the image subtraction baseline."""

import numpy as np
from scipy import ndimage
import torch
from torch import nn

from .development import digest


class TransientNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(4, 12, 5, padding=2), nn.SiLU(), nn.MaxPool2d(2),
                                      nn.Conv2d(12, 24, 3, padding=1), nn.SiLU(), nn.AdaptiveAvgPool2d(3),
                                      nn.Flatten(), nn.Linear(216, 32), nn.SiLU(), nn.Linear(32, 1))

    def forward(self, image):
        return self.features(image).squeeze(1)


def stamps(delta, rows):
    reference, science, difference, variance, valid, _ = delta
    step = np.arange(25) - 12
    scale = np.sqrt(variance)
    channels = [np.arcsinh(np.divide(values, scale, out=np.zeros_like(values), where=valid))
                for values in (reference, science, difference)]
    result = []
    for row in rows:
        yy, xx = np.meshgrid(row["y_px"] + step, row["x_px"] + step, indexing="ij")
        coords = np.stack((yy, xx))
        coverage = ndimage.map_coordinates(valid, coords, order=0, mode="constant", cval=0)
        planes = [np.clip(ndimage.map_coordinates(plane, coords, order=1, mode="constant", cval=0), -8, 8)
                  * coverage for plane in channels]
        result.append(np.stack((*planes, coverage), dtype=np.float32))
    return np.asarray(result, dtype=np.float32)


def load_classifier(path, device):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state.get("kind") != "hermae-transient-v1":
        raise ValueError("Expected a Hermae transient classifier checkpoint.")
    model = TransientNet()
    model.load_state_dict(state["model"])
    return model.to(device).eval(), state, digest(path)


@torch.inference_mode()
def classify(model, delta, rows, device):
    result = []
    image = torch.from_numpy(stamps(delta, rows))
    for batch in image.split(128):
        result.extend(model(batch.to(device)).sigmoid().cpu().tolist())
    return result
