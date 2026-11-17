"""Train the practical-specific models from the project root.

Examples:
  python -m scripts.train logic
  python -m scripts.train lenet --quick
  python -m scripts.train transfer --custom-data data/custom_digits
"""
import argparse
from handwrite_ai.config import ensure_dirs
from handwrite_ai import experiments as ex

TASKS = ("logic", "perceptron", "mlp", "optimizers", "compression",
         "regularization", "lenet", "transfer", "lstm", "gan", "all")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="HandWrite AI experiment trainer")
    p.add_argument("task", choices=TASKS)
    p.add_argument("--quick", action="store_true", help="Small dataset, one epoch (smoke test only)")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--train-limit", type=int, default=12000,
                   help="Max MNIST training examples; 0 uses entire split")
    p.add_argument("--test-limit", type=int, default=2000,
                   help="Max MNIST test examples; 0 uses all 10000")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--custom-data", default="data/custom_digits")
    p.add_argument("--pendigits-dir", default=None)
    p.add_argument("--window", type=int, default=3)
    a = p.parse_args(argv)
    if a.quick:
        a.train_limit = 2000
        a.test_limit = 500
        a.epochs = a.epochs or 1
    if a.train_limit < 0 or a.test_limit < 0:
        p.error("Dataset limits must be nonnegative")
    if a.epochs is not None and a.epochs < 1:
        p.error("Epochs must be positive")
    if a.batch_size < 1:
        p.error("Batch size must be positive")
    a.train_limit = a.train_limit or None
    a.test_limit = a.test_limit or None
    return a


def run(task, args):
    epochs = args.epochs
    common = {"train_limit": args.train_limit, "test_limit": args.test_limit}
    batch = args.batch_size
    if task == "logic":
        return ex.run_logic()
    if task == "perceptron":
        return ex.run_perceptron(**common)
    if task in ("mlp", "lenet"):
        return ex.run_classifier(task, epochs=epochs or 5, batch_size=batch, **common)
    if task == "optimizers":
        return ex.run_optimizers(epochs=epochs or 5, batch_size=batch, **common)
    if task == "regularization":
        return ex.run_regularization(epochs=epochs or 8, batch_size=batch, **common)
    if task == "compression":
        return ex.run_compression(epochs=epochs or 8, batch_size=batch,
                                  latent_dim=args.latent_dim, **common)
    if task == "transfer":
        return ex.run_transfer(args.custom_data, epochs=epochs or 4,
                               batch_size=min(batch, 32))
    if task == "lstm":
        return ex.run_lstm(args.pendigits_dir, epochs=epochs or 12,
                           batch_size=batch, window=args.window)
    if task == "gan":
        return ex.run_gan(train_limit=args.train_limit, epochs=epochs or 10,
                          batch_size=batch)
    raise ValueError(task)


def main(argv=None):
    args = parse_args(argv)
    ensure_dirs()
    jobs = TASKS[:-1] if args.task == "all" else (args.task,)
    for task in jobs:
        print(f"\n{'='*64}\nEXPERIMENT: {task.upper()}\n{'='*64}")
        try:
            run(task, args)
        except (FileNotFoundError, RuntimeError, ValueError) as exc:
            if args.task == "all" and task == "transfer":
                print(f"SKIPPED transfer: {exc}")
                continue
            raise


if __name__ == "__main__":
    main()
