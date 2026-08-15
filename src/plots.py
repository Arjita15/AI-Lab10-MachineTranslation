"""Generates the comparison plots used in results/plots/."""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from src import config
from src.utils import load_json

DISPLAY_NAMES = {
    "rnn_seq2seq": "Vanilla RNN",
    "encoder_decoder": "GRU Enc-Dec",
    "additive_attention": "Additive Attn",
    "multiplicative_attention": "Multiplicative Attn",
}
COLORS = {
    "rnn_seq2seq": "#9aa5b1",
    "encoder_decoder": "#5b8def",
    "additive_attention": "#2ca58d",
    "multiplicative_attention": "#e8871e",
}


def plot_loss_curves():
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for name in config.MODEL_NAMES:
        path = os.path.join(config.METRICS_DIR, f"{name}_history.json")
        if not os.path.exists(path):
            continue
        hist = load_json(path)
        epochs = range(1, len(hist["train_loss"]) + 1)
        axes[0].plot(epochs, hist["train_loss"], marker="o", ms=3, label=DISPLAY_NAMES[name], color=COLORS[name])
        axes[1].plot(epochs, hist["val_loss"], marker="o", ms=3, label=DISPLAY_NAMES[name], color=COLORS[name])

    for ax, title in zip(axes, ["Training loss", "Validation loss"]):
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Cross-entropy loss")
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle("Training / validation loss across the 4 architectures")
    fig.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "loss_curves.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


def plot_bleu_comparison():
    csv_path = os.path.join(config.METRICS_DIR, "comparison.csv")
    if not os.path.exists(csv_path):
        print("No comparison.csv found, run evaluate first")
        return
    df = pd.read_csv(csv_path)
    df["short"] = df["model"].map(DISPLAY_NAMES)
    df = df.sort_values("bleu")

    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = [COLORS[m] for m in df["model"]]
    bars = ax.barh(df["short"], df["bleu"], color=colors)
    ax.set_xlabel("BLEU score (test set)")
    ax.set_title("Test-set BLEU by architecture")
    ax.grid(axis="x", alpha=0.3)
    for bar, val in zip(bars, df["bleu"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height() / 2, f"{val:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    out_path = os.path.join(config.PLOTS_DIR, "bleu_comparison.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_loss_curves()
    plot_bleu_comparison()
