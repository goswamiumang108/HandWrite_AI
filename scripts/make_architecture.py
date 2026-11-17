"""Regenerate the project's architecture illustration (matplotlib only)."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "assets" / "architecture.png"

fig, ax = plt.subplots(figsize=(12.5, 6.5))
fig.patch.set_facecolor("#0d1928")
ax.set_facecolor("#0d1928")
ax.set(xlim=(0, 12.5), ylim=(0, 6.5))
ax.axis("off")

def block(x, y, w, h, title, desc, color="#1b3449"):
    rect = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                          linewidth=1.0, edgecolor="#47647d", facecolor=color)
    ax.add_patch(rect)
    ax.text(x+w/2, y+h*.67, title, ha="center", va="center",
            color="#ecf7fa", fontsize=11, weight="bold")
    ax.text(x+w/2, y+h*.3, desc, ha="center", va="center",
            color="#abc9d8", fontsize=8.1)

def arrow(x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                  mutation_scale=13, lw=1.5, color="#42c7bf"))

ax.text(6.25, 6.15, "HandWrite AI  |  System Architecture", color="#f3fafc",
        ha="center", va="center", fontsize=21, weight="bold")
block(3.65, 5.0, 5.2, .82, "Streamlit application", "Upload · Recognize · Compare · Compress · Generate", "#23404f")
arrow(6.25, 5, 6.25, 4.78)
block(3.65, 4.0, 5.2, .75, "Preprocessing & data loading", "MNIST · Custom digit images · UCI PenDigits", "#203b4d")
for x in (0.65, 3.75, 6.85, 9.95):
    arrow(6.25, 4, x+1.06, 3.58)
block(.35, 2.55, 2.6, 1.03, "Neural foundations", "Logic gates · Perceptron · MLP")
block(3.45, 2.55, 2.6, 1.03, "Performance lab", "Optimizers · Dropout · BatchNorm")
block(6.55, 2.55, 2.6, 1.03, "Representation", "PCA · Autoencoder · LeNet-5")
block(9.65, 2.55, 2.6, 1.03, "Advanced learning", "VGG16 · LSTM · GAN")
for x in (1.65, 4.75, 7.85, 10.95):
    arrow(x, 2.55, 6.25, 2.0)
block(3.65, 1.2, 5.2, .74, "Saved models & measured results", ".keras · .json · .npz · plots", "#23404f")
arrow(6.25, 1.2, 6.25, .98)
block(3.65, .17, 5.2, .77, "Analysis dashboard", "Predictions · Confusion matrices · Reconstruction · Generated samples", "#225a59")
fig.tight_layout(pad=.3)
OUTPUT.parent.mkdir(exist_ok=True, parents=True)
fig.savefig(OUTPUT, dpi=170, facecolor=fig.get_facecolor())
print(OUTPUT)
