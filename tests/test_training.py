"""Check that early stopping restores an independent copy of the best weights."""

from importlib.util import find_spec
import unittest

if find_spec("torch"):
    import torch
    from hermae.training import BestCheckpoint
else:
    torch = None


@unittest.skipIf(torch is None, "requires the optional ML dependencies")
class TrainingTest(unittest.TestCase):
    def test_plateau_reduces_rate_and_restores_best_weights(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        selector = BestCheckpoint(patience=3)
        self.assertFalse(selector.observe(model, optimizer, (4, .8)))
        with torch.no_grad():
            model.weight.fill_(3.)
        self.assertFalse(selector.observe(model, optimizer, (4, .7)))
        self.assertEqual(optimizer.param_groups[0]["lr"], .005)
        self.assertFalse(selector.observe(model, optimizer, (3, .9)))
        self.assertTrue(selector.observe(model, optimizer, (3, .9)))
        self.assertEqual(selector.best_epoch, 1)
        self.assertEqual(selector.restore(model).weight.item(), 2.)
