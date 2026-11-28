"""Check the displayed batch against real model outputs without opening a window."""

from importlib.util import find_spec
from queue import Queue
from threading import Event
from types import SimpleNamespace
import unittest

import numpy as np

from hermae.training_preview import TrainingPreview, captured_forward

if find_spec("torch"):
    import torch
else:
    torch = None


@unittest.skipIf(torch is None, "requires the optional ML dependencies")
class PreviewTest(unittest.TestCase):
    def test_forward_capture_preserves_logits_and_gradients(self):
        encoder = torch.nn.Conv2d(1, 3, 3, padding=1)
        model = torch.nn.Sequential(encoder, torch.nn.ReLU(), torch.nn.Conv2d(3, 1, 1))
        images = torch.randn(3, 1, 8, 8)
        expected = model(images)
        expected.sum().backward()
        gradient = encoder.weight.grad.clone()
        model.zero_grad()
        logits, spatial = captured_forward(model, encoder, images)
        logits.sum().backward()
        torch.testing.assert_close(logits, expected)
        torch.testing.assert_close(encoder.weight.grad, gradient)
        self.assertEqual(tuple(spatial.shape), (3, 8, 8))
        self.assertFalse(spatial.requires_grad)

    def test_first_epoch_queues_every_image_and_later_epochs_can_sample(self):
        preview = TrainingPreview.__new__(TrainingPreview)
        preview.mode, preview.last_preview = "first-pass", float("-inf")
        preview.frames, preview.closed = Queue(), Event()
        preview.process = SimpleNamespace(is_alive=lambda: True)
        pixels = torch.stack((torch.zeros(8, 8), torch.ones(8, 8)))
        panels = (("Score", pixels, "heat"),)
        preview.show("test", 1, 150, panels, ["first", "second"])
        _, epoch, total, frames, details = preview.frames.get_nowait()
        self.assertEqual((epoch, total, details), (1, 150, ["first", "second"]))
        self.assertEqual(len(frames[0]), 2)
        self.assertTrue(np.all(frames[0][1][1][:, :, 0] == 255))
        preview.last_preview = float("-inf")
        preview.show("test", 2, 150, panels, ["first", "second"])
        self.assertEqual(preview.frames.get_nowait()[-1], ["first"])
        preview.mode = "all"
        preview.show("test", 3, 150, panels, ["first", "second"])
        self.assertEqual(preview.frames.get_nowait()[-1], ["first", "second"])
        preview.closed.set()
        self.assertFalse(preview.active(1))
