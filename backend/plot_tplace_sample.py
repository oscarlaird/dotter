import importlib.util
import json
import math
import pathlib
import subprocess

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = pathlib.Path(__file__).resolve().parent
TPLACE_PATH = ROOT / "tplace_scipy_fast.py"
RUST_BINARY = ROOT.parent / "bayesian/target/release/plot_data"
OUTPUT_PATH = ROOT / "tplace_sample_plot.png"

spec = importlib.util.spec_from_file_location("tplace", TPLACE_PATH)
tplace = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tplace)


SEED = 0
P = 1.0
K = 300
SIGMA = 0.020
F = tplace.DEFAULT_F
PLOT_POINTS = 2000


def sample_weights():
    generator = torch.Generator().manual_seed(SEED)
    weight_logits = torch.randn(K, generator=generator)
    return torch.softmax(weight_logits, dim=0)


def run_rust_optimizer(weights: torch.Tensor) -> dict:
    payload = json.dumps({
        "weights": weights.tolist(),
        "sigma": SIGMA,
        "period": P,
        "f": F,
        "iter_counts": [10, 25, 100],
    })
    result = subprocess.run(
        [str(RUST_BINARY)],
        input=payload,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(result.stdout)


def weighted_bells(phases: np.ndarray, weights: torch.Tensor, x: torch.Tensor):
    phases_t = torch.as_tensor(phases, dtype=x.dtype, device=x.device)
    dists = x[None, :] - phases_t[:, None]
    inv_sigma_sq = 1 / (2 * SIGMA**2)
    log_norm = -0.5 * math.log(2 * math.pi * SIGMA**2)
    densities = (
        torch.exp(-dists**2 * inv_sigma_sq + log_norm)
        + torch.exp(-(dists - P) ** 2 * inv_sigma_sq + log_norm)
        + torch.exp(-(dists + P) ** 2 * inv_sigma_sq + log_norm)
    )
    weighted = weights[:, None] * densities
    total = weighted.sum(dim=0)
    return weighted, total


def differential_entropy_from_pdf(pdf: torch.Tensor, x: torch.Tensor):
    return float(-torch.trapezoid(pdf * pdf.log(), x))


def main():
    weights = sample_weights()
    params = tplace.TPlaceParams(weights=weights, sigma=SIGMA, P=P, F=F)
    noise_entropy = 0.5 * math.log(2 * math.pi * math.e * SIGMA**2)

    rust_data = run_rust_optimizer(weights)
    initial_phases = np.array(rust_data["initial_phases"])
    optimized_by_iter = {e["max_iter"]: (np.array(e["phases"]), e["loss"]) for e in rust_data["optimized"]}

    x = torch.linspace(0, P, PLOT_POINTS, dtype=weights.dtype, device=weights.device)
    scenarios = [
        ("Initial constant", initial_phases, None),
        ("After optimize (max_iter=10)", *optimized_by_iter[10]),
        ("After optimize (max_iter=25)", *optimized_by_iter[25]),
        ("After optimize (max_iter=100)", *optimized_by_iter[100]),
    ]

    order = torch.argsort(weights, descending=True)
    colors = plt.cm.nipy_spectral(torch.linspace(0.02, 0.98, K).numpy())
    fig, axes = plt.subplots(2, 2, figsize=(15, 10), sharex=True, sharey=True, constrained_layout=True)

    for ax, (title, phases, rust_loss) in zip(axes.flatten(), scenarios):
        weighted, total = weighted_bells(phases, weights, x)
        objective = float(tplace.J(np.asarray(phases), params)) if rust_loss is None else rust_loss
        differential_entropy = differential_entropy_from_pdf(total, x)
        usable_entropy = differential_entropy - noise_entropy
        ax.stackplot(x.numpy(), weighted[order].numpy(), colors=colors, alpha=0.95, linewidth=0)
        ax.plot(x.numpy(), total.numpy(), color="black", linewidth=2.2, label="total_density")
        ax.set_title(
            f"{title}\n"
            f"frequency-domain loss={objective:.4f}\n"
            f"differential entropy={differential_entropy:.4f}\n"
            f"usable entropy={usable_entropy:.4f}\n"
            f"min phase={float(np.min(phases)):.5f}, max phase={float(np.max(phases)):.5f}"
        )
        ax.set_xlim(0.0, P)
        ax.set_ylabel("weighted density")
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")

    axes[1, 0].set_xlabel("x")
    axes[1, 1].set_xlabel("x")
    fig.suptitle(
        f"tplace sample | seed={SEED}, K={K}, P={P}, sigma={SIGMA}, F={F}\n"
        "Real-space wrapped Gaussian visualization\n"
        "Stacked weighted components ordered by descending weight",
        fontsize=14,
    )
    fig.savefig(OUTPUT_PATH, dpi=180)
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
