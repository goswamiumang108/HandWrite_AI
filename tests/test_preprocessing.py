import pytest
from PIL import Image, ImageDraw
import numpy as np
from handwrite_ai.preprocessing import prepare_digit


def test_black_ink_on_white_background():
    image = Image.new("L", (100, 100), 255)
    ImageDraw.Draw(image).line([(30, 10), (70, 80)], fill=0, width=8)
    x = prepare_digit(image)
    assert x.shape == (28, 28)
    assert x.dtype == np.float32
    assert 0 <= x.min() <= x.max() <= 1
    assert x.sum() > 10
    assert x[0].sum() == 0


def test_white_ink_on_black_background():
    image = Image.new("L", (100, 100), 0)
    ImageDraw.Draw(image).line([(30, 10), (70, 80)], fill=255, width=8)
    x = prepare_digit(image)
    assert x.shape == (28, 28)
    assert x.sum() > 10


def test_blank_is_rejected():
    with pytest.raises(ValueError, match="No visible digit"):
        prepare_digit(Image.new("L", (28, 28), 255))
