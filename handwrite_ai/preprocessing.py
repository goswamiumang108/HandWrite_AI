"""Center handwriting for MNIST-style inference, independent of TensorFlow."""
from PIL import Image, ImageOps
import numpy as np


def prepare_digit(image: Image.Image, target_size: int = 28, digit_size: int | None = None) -> np.ndarray:
    """Return (H,W) float32 array in [0,1], bright ink on a dark background.

    A white-page/black-ink image is inverted. Detect the foreground bbox and
    preserve its aspect ratio before centering. The background is estimated
    from the image's border, not the entire image.
    """
    if target_size < 8:
        raise ValueError("target_size must be at least 8")
    if digit_size is None:
        digit_size = int(round(target_size * 0.72))
    if not (1 <= digit_size <= target_size):
        raise ValueError("digit_size must be within the output size")
    gray = ImageOps.exif_transpose(image).convert("L")
    arr = np.asarray(gray, dtype=np.uint8)
    border = np.concatenate((arr[0], arr[-1], arr[:, 0], arr[:, -1]))
    if float(np.median(border)) > 127:
        arr = 255 - arr
    # Suppress weak backgrounds from scanned paper.
    arr = np.where(arr >= 28, arr, 0).astype(np.uint8)
    ys, xs = np.where(arr > 0)
    if len(xs) == 0:
        raise ValueError("No visible digit found. Upload a clear handwritten numeral.")
    cropped = Image.fromarray(arr[ys.min():ys.max()+1, xs.min():xs.max()+1])
    scale = min(digit_size / cropped.width, digit_size / cropped.height)
    resized = cropped.resize((max(1, round(cropped.width * scale)),
                              max(1, round(cropped.height * scale))), Image.Resampling.LANCZOS)
    canvas = Image.new("L", (target_size, target_size), 0)
    canvas.paste(resized, ((target_size - resized.width)//2,
                           (target_size - resized.height)//2))
    return np.asarray(canvas, dtype=np.float32) / 255.0
