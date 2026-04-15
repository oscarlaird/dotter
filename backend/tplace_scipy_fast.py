import math
import time
from typing import NamedTuple

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import torch


FTYPE = np.float64


class TPlaceParams(NamedTuple):
    weights: torch.Tensor
    sigma: float
    P: float
    F: int


BLOCK_SIZE = 8
DEFAULT_MAX_ITER = 25
DEFAULT_F = 16


def _as_numpy_1d(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(FTYPE, copy=False)
    return np.asarray(values, dtype=FTYPE)


def loss_and_grad_for_phases(phases, params: TPlaceParams):
    phases = _as_numpy_1d(phases)
    weights = _as_numpy_1d(params.weights)

    ns = np.arange(0, params.F + 1, dtype=FTYPE)
    alpha = (2 * math.pi / params.P) * ns
    base_mag2 = np.exp(
        -2 * math.log(params.P) - 4 * math.pi**2 * ns**2 * params.sigma**2 / params.P**2
    )

    theta = (2 * math.pi / params.P) * phases
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    block_size = min(BLOCK_SIZE, params.F)
    block_cos = np.empty((phases.size, block_size), dtype=FTYPE)
    block_sin = np.empty((phases.size, block_size), dtype=FTYPE)
    if block_size > 0:
        block_cos[:, 0] = cos_theta
        block_sin[:, 0] = sin_theta
        for m in range(1, block_size):
            prev_cos = block_cos[:, m - 1]
            prev_sin = block_sin[:, m - 1]
            block_cos[:, m] = prev_cos * cos_theta - prev_sin * sin_theta
            block_sin[:, m] = prev_sin * cos_theta + prev_cos * sin_theta

    a = np.empty(params.F + 1, dtype=FTYPE)
    b = np.empty(params.F + 1, dtype=FTYPE)
    a[0] = weights.sum()
    b[0] = 0.0
    loss = base_mag2[0] * (a[0] * a[0])

    cos_base = np.ones_like(theta)
    sin_base = np.zeros_like(theta)
    grad = np.zeros_like(theta)

    for start in range(1, params.F + 1, BLOCK_SIZE):
        width = min(BLOCK_SIZE, params.F - start + 1)
        cos_f = block_cos[:, :width]
        sin_f = block_sin[:, :width]

        cos_batch = cos_base[:, None] * cos_f - sin_base[:, None] * sin_f
        sin_batch = sin_base[:, None] * cos_f + cos_base[:, None] * sin_f

        idx = np.arange(start, start + width)
        a_vals = weights @ cos_batch
        b_vals = weights @ sin_batch
        a[idx] = a_vals
        b[idx] = b_vals

        loss += 2.0 * np.sum(base_mag2[idx] * (a_vals * a_vals + b_vals * b_vals))

        coeff = 4.0 * alpha[idx] * base_mag2[idx]
        grad += weights * (cos_batch @ (coeff * b_vals) - sin_batch @ (coeff * a_vals))

        cos_base = cos_batch[:, -1].copy()
        sin_base = sin_batch[:, -1].copy()

    return loss, grad


def constant_phases(count: int, period: float):
    step = period / count
    return (np.arange(count, dtype=FTYPE) + 0.5) * step


def J(phases, params: TPlaceParams):
    return loss_and_grad_for_phases(phases, params)[0]


def optimize(params: TPlaceParams, initial_phases, *, max_iter: int = DEFAULT_MAX_ITER):
    initial_phases = _as_numpy_1d(initial_phases)
    start_time = time.perf_counter()
    phases, _, info = fmin_l_bfgs_b(
        lambda phases: loss_and_grad_for_phases(phases, params),
        x0=initial_phases,
        maxiter=max_iter,
    )
    elapsed = time.perf_counter() - start_time
    print(
        "LBFGS converged in %d iterations (%d closure evaluations) over %.3fs"
        % (info["nit"], info["funcalls"], elapsed)
    )
    return phases


if __name__ == "__main__":
    print(1)
    params = TPlaceParams(
        weights=torch.tensor([0.5, 0.25, 0.25]),
        sigma=0.1,
        P=1.0,
        F=DEFAULT_F,
    )
    initial_phases = constant_phases(3, params.P)
    print(f"Initial phases: {initial_phases}")
    print(f"Initial loss: {J(initial_phases, params)}")
    phases = optimize(params, initial_phases)
    print(f"Optimized phases: {phases}")
    print(f"Optimized loss: {J(phases, params)}")
