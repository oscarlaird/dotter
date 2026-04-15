import importlib.util
import pathlib
import statistics
import time

import torch


ROOT = pathlib.Path(__file__).resolve().parent
TPLACE_PATH = ROOT / "tplace.py"
spec = importlib.util.spec_from_file_location("tplace", TPLACE_PATH)
tplace = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(tplace)


def sample_case(k: int, p: float, sigma: float, n: int, generator: torch.Generator):
    weight_logits = torch.randn(k, generator=generator)
    weights = torch.softmax(weight_logits, dim=0)
    initial_phases = tplace.constant_phases(k, p)

    params = tplace.TPlaceParams(weights=weights, sigma=sigma, P=p, F=n)
    return params, initial_phases


def run_benchmark(
    *,
    k: int = 200,
    n: int = 1000,
    p: float = 1.0,
    sigma: float = 0.1,
    trials: int = 5,
    seed: int = 0,
):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(seed)

    warmup_params, warmup_initial_phases = sample_case(k, p, sigma, n, generator)
    tplace.optimize(warmup_params, warmup_initial_phases)

    trial_times = []
    loss_deltas = []
    for trial_idx in range(trials):
        params, initial_phases = sample_case(k, p, sigma, n, generator)
        initial_loss = float(tplace.J(initial_phases, params))

        start = time.perf_counter()
        phases = tplace.optimize(params, initial_phases)
        elapsed = time.perf_counter() - start

        final_loss = float(tplace.J(phases, params))
        loss_delta = final_loss - initial_loss

        trial_times.append(elapsed)
        loss_deltas.append(loss_delta)
        print(
            f"trial={trial_idx + 1} elapsed_s={elapsed:.6f} "
            f"loss_delta={loss_delta:.6f}"
        )

    print("SUMMARY")
    print(f"K={k} F={n} sigma={sigma} trials={trials} seed={seed}")
    print(f"mean_elapsed_s={statistics.mean(trial_times):.6f}")
    print(f"median_elapsed_s={statistics.median(trial_times):.6f}")
    print(f"min_elapsed_s={min(trial_times):.6f}")
    print(f"max_elapsed_s={max(trial_times):.6f}")
    print(f"mean_loss_delta={statistics.mean(loss_deltas):.6f}")


if __name__ == "__main__":
    run_benchmark()
