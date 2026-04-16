import math
import time
from typing import NamedTuple

import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import torch


class TPlaceParams(NamedTuple):
    weights: torch.Tensor
    sigma: float
    P: float
    F: int


def _as_numpy_1d(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(np.float64, copy=False)
    return np.asarray(values, dtype=np.float64)


def _advance_trig(cos_n, sin_n, cos_theta, sin_theta, next_cos_n, next_sin_n, tmp1, tmp2):
    np.multiply(cos_n, cos_theta, out=tmp1)
    np.multiply(sin_n, sin_theta, out=tmp2)
    np.subtract(tmp1, tmp2, out=next_cos_n)
    np.multiply(sin_n, cos_theta, out=tmp1)
    np.multiply(cos_n, sin_theta, out=tmp2)
    np.add(tmp1, tmp2, out=next_sin_n)
    return next_cos_n, next_sin_n


def _compute_mode_sums(weights, cos_theta, sin_theta, num_modes, cos_n, sin_n, next_cos_n, next_sin_n, tmp1, tmp2):
    a = np.empty(num_modes, dtype=np.float64)
    b = np.empty(num_modes, dtype=np.float64)
    a[0] = weights.sum()
    b[0] = 0.0
    for n in range(1, num_modes):
        _advance_trig(cos_n, sin_n, cos_theta, sin_theta, next_cos_n, next_sin_n, tmp1, tmp2)
        cos_n, next_cos_n = next_cos_n, cos_n
        sin_n, next_sin_n = next_sin_n, sin_n
        a[n] = weights @ cos_n
        b[n] = weights @ sin_n
    return a, b


def _compute_phase_grad(weights, alpha, base_mag2, a, b, cos_theta, sin_theta, cos_n, sin_n, next_cos_n, next_sin_n, tmp1, tmp2):
    grad = np.zeros_like(cos_theta)
    grad_term = np.empty_like(cos_theta)
    for n in range(1, len(alpha)):
        _advance_trig(cos_n, sin_n, cos_theta, sin_theta, next_cos_n, next_sin_n, tmp1, tmp2)
        cos_n, next_cos_n = next_cos_n, cos_n
        sin_n, next_sin_n = next_sin_n, sin_n
        np.multiply(cos_n, b[n], out=tmp1)
        np.multiply(sin_n, a[n], out=tmp2)
        np.subtract(tmp1, tmp2, out=grad_term)
        np.multiply(grad_term, weights, out=grad_term)
        grad_term *= 4.0 * alpha[n] * base_mag2[n]
        np.add(grad, grad_term, out=grad)
    return grad


def loss_and_grad_for_phases(phases, params: TPlaceParams):
    phases = _as_numpy_1d(phases)
    weights = _as_numpy_1d(params.weights)

    ns = np.arange(0, params.F + 1, dtype=np.float64)
    alpha = (2 * math.pi / params.P) * ns
    base_mag2 = np.exp(
        -2 * math.log(params.P) - 4 * math.pi**2 * ns**2 * params.sigma**2 / params.P**2
    )

    theta = (2 * math.pi / params.P) * phases
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    cos_n = np.ones_like(theta)
    sin_n = np.zeros_like(theta)
    next_cos_n = np.empty_like(theta)
    next_sin_n = np.empty_like(theta)
    tmp1 = np.empty_like(theta)
    tmp2 = np.empty_like(theta)
    mode_tmp = np.empty(params.F + 1, dtype=np.float64)
    a, b = _compute_mode_sums(
        weights, cos_theta, sin_theta, params.F + 1, cos_n, sin_n, next_cos_n, next_sin_n, tmp1, tmp2
    )

    interference_mag2 = np.empty_like(a)
    np.multiply(a, a, out=interference_mag2)
    np.multiply(b, b, out=mode_tmp)
    np.add(interference_mag2, mode_tmp, out=interference_mag2)
    loss = base_mag2[0] * interference_mag2[0] + 2.0 * np.sum(base_mag2[1:] * interference_mag2[1:])

    cos_n = np.ones_like(theta)
    sin_n = np.zeros_like(theta)
    grad = _compute_phase_grad(
        weights, alpha, base_mag2, a, b, cos_theta, sin_theta, cos_n, sin_n, next_cos_n, next_sin_n, tmp1, tmp2
    )

    return loss, grad


def constant_phases(count: int, period: float):
    step = period / count
    return (np.arange(count, dtype=np.float64) + 0.5) * step


def J(phases, params: TPlaceParams):
    return loss_and_grad_for_phases(phases, params)[0]


def optimize(params: TPlaceParams, initial_phases, *, max_iter: int = 100):
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
        F=10,
    )
    initial_phases = constant_phases(3, params.P)
    print(f"Initial phases: {initial_phases}")
    print(f"Initial loss: {J(initial_phases, params)}")
    phases = optimize(params, initial_phases)
    print(f"Optimized phases: {phases}")
    print(f"Optimized loss: {J(phases, params)}")
