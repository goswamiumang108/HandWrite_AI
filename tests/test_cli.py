import pytest
from scripts.train import parse_args


def test_quick_uses_limited_data():
    args = parse_args(["lenet", "--quick"])
    assert args.epochs == 1
    assert args.train_limit == 2000
    assert args.test_limit == 500


def test_zero_limit_means_full_split():
    args = parse_args(["mlp", "--train-limit", "0", "--test-limit", "0"])
    assert args.train_limit is None
    assert args.test_limit is None


def test_invalid_training_args():
    with pytest.raises(SystemExit):
        parse_args(["lenet", "--epochs", "0"])
