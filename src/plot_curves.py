import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def smooth(values, k=10):
    out = []
    for i in range(len(values)):
        w = values[max(0, i - k + 1): i + 1]
        out.append(sum(w) / len(w))
    return out


def main():
    files = sorted(Path("results").glob("*_history.json"))
    if not files:
        print("No history files found in results/")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    for f in files:
        h = json.load(open(f))
        name = f.stem.replace("_history", "")
        steps, losses = zip(*h["train"])
        ax1.plot(steps, smooth(list(losses)), label=name)
        vs, vl = zip(*h["val"])
        ax2.plot(vs, vl, marker="o", label=name)
    ax1.set_title("Training loss (smoothed)")
    ax2.set_title("Validation loss (step 0 = pretrained baseline)")
    for ax in (ax1, ax2):
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("loss")
        ax.legend()
        ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/curves.png", dpi=150)
    print("Saved results/curves.png")


if __name__ == "__main__":
    main()