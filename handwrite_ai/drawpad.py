"""Bundled no-CDN Streamlit canvas component for handwriting input."""
from pathlib import Path
import base64
import io
from PIL import Image


def draw_digit(key: str):
    import streamlit.components.v1 as components
    path = Path(__file__).resolve().parent.parent / "components" / "digit_pad"
    pad = components.declare_component("handwrite_digit_pad", path=str(path))
    data_url = pad(key=key, default="")
    if not data_url or not isinstance(data_url, str):
        return None
    prefix = "data:image/png;base64,"
    if not data_url.startswith(prefix):
        return None
    raw = base64.b64decode(data_url[len(prefix):], validate=True)
    return Image.open(io.BytesIO(raw)).convert("RGB")
