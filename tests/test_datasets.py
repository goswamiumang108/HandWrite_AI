import numpy as np
import pytest
from handwrite_ai.datasets import sliding_windows


def test_sliding_windows_with_next_targets():
    seq = np.arange(2*8*2, dtype="float32").reshape(2, 8, 2)
    x, y = sliding_windows(seq, length=3)
    assert x.shape == (10, 3, 2)
    assert y.shape == (10, 2)
    np.testing.assert_array_equal(x[0], seq[0, 0:3])
    np.testing.assert_array_equal(y[0], seq[0, 3])
    np.testing.assert_array_equal(x[2], seq[0, 1:4])


def test_invalid_sliding_window():
    with pytest.raises(ValueError):
        sliding_windows(np.zeros((1, 8, 2)), 8)
