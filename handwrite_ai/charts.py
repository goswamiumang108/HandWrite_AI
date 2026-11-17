"""Reusable readable matplotlib charts (saves nothing unless requested)."""
import matplotlib.pyplot as plt
import numpy as np


def line_plot(series: dict[str, list], title: str, ylabel: str):
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, values in series.items():
        if values:
            ax.plot(range(1, len(values)+1), values, marker="o", label=name)
    ax.set(xlabel="Epoch", ylabel=ylabel, title=title)
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    return fig


def digit_grid(images: np.ndarray, title: str = "Digit samples", n: int = 16):
    images = np.asarray(images)[:n]
    cols = 4
    rows = int(np.ceil(len(images) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(7, 1.8 * rows), squeeze=False)
    for ax in axs.ravel():
        ax.axis("off")
    for ax, img in zip(axs.ravel(), images):
        ax.imshow(img.squeeze(), cmap="gray", vmin=0, vmax=1)
    fig.suptitle(title)
    fig.tight_layout()
    return fig
