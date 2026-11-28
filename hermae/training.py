"""Common optimization settings and development checkpoint selection."""

import math


DEFAULT_EPOCHS = 150
GALAXY_CROPS_PER_FIELD = 64
RANKER_VIEWS_PER_FIELD = 2
TRANSIENT_VIEWS_PER_STAMP = 4
UV_PATCHES_PER_IMAGE = 384


def orient_stamps(images, orientation):
    """Apply a square-image rotation or reflection to every channel together."""
    if orientation % 4:
        images = images.rot90(orientation % 4, (-2, -1))
    return images.flip((-1,)) if orientation >= 4 else images


def optimization_arguments(parser, *, learning_rate, patience):
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="number of epochs to complete")
    parser.add_argument("--learning-rate", type=float, default=learning_rate, help="initial AdamW learning rate")
    parser.add_argument("--weight-decay", type=float, default=.01)
    parser.add_argument("--patience", type=int, default=patience,
                        help="validation plateau window for learning-rate reductions")


def validate_optimization(parser, args):
    if args.epochs < 1 or args.patience < 1:
        parser.error("Epochs and patience must be positive.")
    if not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        parser.error("Learning rate must be positive and finite.")
    if not math.isfinite(args.weight_decay) or args.weight_decay < 0:
        parser.error("Weight decay must be nonnegative and finite.")


class BestCheckpoint:
    """Keep the highest development score and lower the rate during plateaus."""

    def __init__(self, patience, rates=None):
        self.patience = patience
        self.rates = rates
        self.best_epoch = 0
        self.best_score = None
        self.weights = None
        self.history = []

    def start_epoch(self, optimizer):
        if self.rates is not None:
            for group in optimizer.param_groups:
                group["lr"] = self.rates[len(self.history)]

    def observe(self, model, optimizer, score, **metrics):
        if not all(math.isfinite(value) for value in score):
            raise ValueError("Training produced a nonfinite validation score.")
        epoch = len(self.history) + 1
        self.history.append(dict(epoch=epoch, learning_rate=optimizer.param_groups[0]["lr"], **metrics))
        if self.rates is not None:
            self.best_epoch = epoch
            return
        if self.best_score is None or score > self.best_score:
            self.best_score = score
            self.best_epoch = epoch
            self.weights = {name: value.detach().to("cpu", copy=True) for name, value in model.state_dict().items()}
        stale = epoch - self.best_epoch
        if stale and stale % max(1, self.patience // 3) == 0:
            for group in optimizer.param_groups:
                group["lr"] = max(group["lr"] * .5, self.history[0]["learning_rate"] * .01)

    def restore(self, model):
        if self.rates is not None:
            print(f"Completed all {self.best_epoch} scheduled epochs", flush=True)
            return model.eval()
        model.load_state_dict(self.weights)
        print(f"Selected epoch {self.best_epoch} of {len(self.history)}", flush=True)
        return model.eval()
