"""Immediately runnable 8x8 recognition demonstration (not the MNIST model)."""
from functools import lru_cache
from PIL import Image
from .preprocessing import prepare_digit
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


@lru_cache(maxsize=1)
def train_demo():
    data = load_digits()
    # Apply the SAME centering/preprocessing used for uploaded drawings.
    x = np.stack([
        prepare_digit(Image.fromarray((img / 16.0 * 255).astype("uint8"), "L"),
                      target_size=8, digit_size=6)
        for img in data.images
    ])
    y = data.target
    xtrain, xtest, ytrain, ytest = train_test_split(
        x.reshape(-1, 64), y, stratify=y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(max_iter=500, random_state=42)
    model.fit(xtrain, ytrain)
    score = accuracy_score(ytest, model.predict(xtest))
    return model, score, data.images, y


def demo_predict(image_array: np.ndarray) -> np.ndarray:
    model = train_demo()[0]
    if image_array.shape != (8, 8):
        raise ValueError("Demo model expects a processed 8x8 image")
    return model.predict_proba(image_array.reshape(1, 64))[0]
