"""Check full schedules and independent copies of the best weights."""

from importlib.util import find_spec
import unittest

if find_spec("torch"):
    import torch
    from hermae.refit import schedule
    from hermae.training import BestCheckpoint, DEFAULT_EPOCHS
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
            progress.observe(model, optimizer, (-i,), train_loss=float(i))
        self.assertEqual([row["learning_rate"] for row in progress.history], [.02, .005])
        self.assertEqual(progress.restore(model).weight.item(), 2.)

    def test_plateau_reduces_rate_and_restores_best_weights(self):
        model = torch.nn.Linear(1, 1, bias=False)
        with torch.no_grad():
            model.weight.fill_(2.)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        selector = BestCheckpoint(patience=3)
        selector.observe(model, optimizer, (4, .8))
        with torch.no_grad():
            model.weight.fill_(3.)
        selector.observe(model, optimizer, (4, .7))
        self.assertEqual(optimizer.param_groups[0]["lr"], .005)
        for _ in range(DEFAULT_EPOCHS - 2):
            selector.observe(model, optimizer, (3, .9))
        self.assertEqual(len(selector.history), 150)
        self.assertEqual(optimizer.param_groups[0]["lr"], .0001)
        self.assertEqual(selector.best_epoch, 1)
        self.assertEqual(selector.restore(model).weight.item(), 2.)

    def test_refit_extends_the_selected_rate_to_the_last_epoch(self):
        prior = dict(seed=42, weight_decay=.01, patience=3)
        model = dict(selected_epoch=2, history=[
            dict(learning_rate=.01), dict(learning_rate=.005), dict(learning_rate=.001),
        ])
        args, rates = schedule(prior, model, "history", "cpu", (), DEFAULT_EPOCHS)
        self.assertEqual(args.epochs, 150)
        self.assertEqual(rates, [.01] + [.005] * 149)
