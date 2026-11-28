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
    def test_fixed_schedule_uses_final_weights_and_assigned_rates(self):
        model = torch.nn.Linear(1, 1, bias=False)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        progress = BestCheckpoint(patience=1, rates=[.02, .005])
        for i in range(2):
            progress.start_epoch(optimizer)
            with torch.no_grad():
                model.weight.fill_(i + 1.)
            self.assertFalse(progress.observe(model, optimizer, (-i,), train_loss=float(i)))
        self.assertEqual([row["learning_rate"] for row in progress.history], [.02, .005])
        self.assertEqual(progress.restore(model).weight.item(), 2.)

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
