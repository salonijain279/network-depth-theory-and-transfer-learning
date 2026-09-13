"""Run the shallow-vs-deep experiment end to end and save the comparison plot + CSV."""
import json
from pathlib import Path

import matplotlib.pyplot as plt

from shallow_vs_deep import run_experiment

OUTPUTS_DIR = Path("outputs")


def main():
    OUTPUTS_DIR.mkdir(exist_ok=True)
    results = run_experiment(repeats=2, epochs=150)

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = {1: "1 hidden layer", 2: "2 hidden layers", 3: "3 hidden layers"}
    colors = {1: "#d9534f", 2: "#f0ad4e", 3: "#5cb85c"}
    for n_layers, data in results.items():
        ax.plot(data["neurons"], data["rmse"], marker="o", label=labels[n_layers], color=colors[n_layers])
    ax.set_xlabel("Neurons per hidden layer")
    ax.set_ylabel("Test RMSE")
    ax.set_title("Shallow vs. Deep Networks on a Compositional Target Function")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUTS_DIR / "shallow_vs_deep_comparison.png", dpi=130, bbox_inches="tight")
    print(f"Saved plot -> {OUTPUTS_DIR / 'shallow_vs_deep_comparison.png'}")

    with open(OUTPUTS_DIR / "shallow_vs_deep_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results -> {OUTPUTS_DIR / 'shallow_vs_deep_results.json'}")


if __name__ == "__main__":
    main()
