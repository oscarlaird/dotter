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
DEFAULT_F = 32


def _as_numpy_1d(values):
    if isinstance(values, torch.Tensor):
        return values.detach().cpu().numpy().astype(FTYPE, copy=False)
    return np.asarray(values, dtype=FTYPE)


class TPlaceState:
    def __init__(self, params: TPlaceParams, phase_count: int):
        self.params = params
        self.phase_count = phase_count
        self.weights = _as_numpy_1d(params.weights)
        self.ns = np.arange(0, params.F + 1, dtype=FTYPE)
        self.alpha = (2 * math.pi / params.P) * self.ns
        self.base_mag2 = np.exp(
            -2 * math.log(params.P) - 4 * math.pi**2 * self.ns**2 * params.sigma**2 / params.P**2
        )
        self.block_size = min(BLOCK_SIZE, params.F)

        self.theta = np.empty(phase_count, dtype=FTYPE)
        self.cos_theta = np.empty(phase_count, dtype=FTYPE)
        self.sin_theta = np.empty(phase_count, dtype=FTYPE)
        self.cos_base = np.empty(phase_count, dtype=FTYPE)
        self.sin_base = np.empty(phase_count, dtype=FTYPE)
        self.tmp1 = np.empty(phase_count, dtype=FTYPE)
        self.tmp2 = np.empty(phase_count, dtype=FTYPE)

        self.block_cos = np.empty((phase_count, self.block_size), dtype=FTYPE)
        self.block_sin = np.empty((phase_count, self.block_size), dtype=FTYPE)
        self.cos_batch = np.empty((phase_count, self.block_size), dtype=FTYPE)
        self.sin_batch = np.empty((phase_count, self.block_size), dtype=FTYPE)
        self.block_tmp = np.empty((phase_count, self.block_size), dtype=FTYPE)

        self.a = np.empty(params.F + 1, dtype=FTYPE)
        self.b = np.empty(params.F + 1, dtype=FTYPE)
        self.grad = np.empty(phase_count, dtype=FTYPE)
        self.mode_tmp1 = np.empty(self.block_size, dtype=FTYPE)
        self.mode_tmp2 = np.empty(self.block_size, dtype=FTYPE)


def _prepare_block_trig(state: TPlaceState):
    if state.block_size == 0:
        return
    state.block_cos[:, 0] = state.cos_theta
    state.block_sin[:, 0] = state.sin_theta
    for m in range(1, state.block_size):
        prev_cos = state.block_cos[:, m - 1]
        prev_sin = state.block_sin[:, m - 1]
        np.multiply(prev_cos, state.cos_theta, out=state.tmp1)
        np.multiply(prev_sin, state.sin_theta, out=state.tmp2)
        np.subtract(state.tmp1, state.tmp2, out=state.block_cos[:, m])
        np.multiply(prev_sin, state.cos_theta, out=state.tmp1)
        np.multiply(prev_cos, state.sin_theta, out=state.tmp2)
        np.add(state.tmp1, state.tmp2, out=state.block_sin[:, m])


def _build_batches(state: TPlaceState, width: int):
    cos_f = state.block_cos[:, :width]
    sin_f = state.block_sin[:, :width]
    cos_batch = state.cos_batch[:, :width]
    sin_batch = state.sin_batch[:, :width]
    block_tmp = state.block_tmp[:, :width]

    np.multiply(state.cos_base[:, None], cos_f, out=cos_batch)
    np.multiply(state.sin_base[:, None], sin_f, out=block_tmp)
    np.subtract(cos_batch, block_tmp, out=cos_batch)

    np.multiply(state.sin_base[:, None], cos_f, out=sin_batch)
    np.multiply(state.cos_base[:, None], sin_f, out=block_tmp)
    np.add(sin_batch, block_tmp, out=sin_batch)
    return cos_batch, sin_batch


def loss_and_grad_for_phases(phases, state: TPlaceState | TPlaceParams):
    phases = _as_numpy_1d(phases)
    if isinstance(state, TPlaceParams):
        state = TPlaceState(state, phases.size)

    np.multiply(phases, 2 * math.pi / state.params.P, out=state.theta)
    np.cos(state.theta, out=state.cos_theta)
    np.sin(state.theta, out=state.sin_theta)
    _prepare_block_trig(state)

    state.a[0] = state.weights.sum()
    state.b[0] = 0.0
    loss = state.base_mag2[0] * (state.a[0] * state.a[0])

    state.cos_base.fill(1.0)
    state.sin_base.fill(0.0)
    state.grad.fill(0.0)

    for start in range(1, state.params.F + 1, BLOCK_SIZE):
        stop = min(start + BLOCK_SIZE, state.params.F + 1)
        width = stop - start
        cos_batch, sin_batch = _build_batches(state, width)

        state.a[start:stop] = state.weights @ cos_batch
        state.b[start:stop] = state.weights @ sin_batch

        loss += 2.0 * np.sum(
            state.base_mag2[start:stop]
            * (state.a[start:stop] * state.a[start:stop] + state.b[start:stop] * state.b[start:stop])
        )

        np.multiply(state.alpha[start:stop], state.base_mag2[start:stop], out=state.mode_tmp1[:width])
        state.mode_tmp1[:width] *= 4.0
        np.multiply(state.mode_tmp1[:width], state.b[start:stop], out=state.mode_tmp1[:width])

        np.multiply(state.alpha[start:stop], state.base_mag2[start:stop], out=state.mode_tmp2[:width])
        state.mode_tmp2[:width] *= 4.0
        np.multiply(state.mode_tmp2[:width], state.a[start:stop], out=state.mode_tmp2[:width])

        np.matmul(cos_batch, state.mode_tmp1[:width], out=state.tmp1)
        np.matmul(sin_batch, state.mode_tmp2[:width], out=state.tmp2)
        np.subtract(state.tmp1, state.tmp2, out=state.tmp1)
        np.multiply(state.tmp1, state.weights, out=state.tmp1)
        np.add(state.grad, state.tmp1, out=state.grad)

        np.copyto(state.cos_base, cos_batch[:, width - 1])
        np.copyto(state.sin_base, sin_batch[:, width - 1])

    return loss, state.grad


def constant_phases(count: int, period: float):
    step = period / count
    return (np.arange(count, dtype=FTYPE) + 0.5) * step


def J(phases, params: TPlaceParams):
    return loss_and_grad_for_phases(phases, params)[0]


def optimize(params: TPlaceParams, initial_phases, *, max_iter: int = DEFAULT_MAX_ITER):
    initial_phases = _as_numpy_1d(initial_phases)
    state = TPlaceState(params, initial_phases.size)
    start_time = time.perf_counter()
    phases, _, info = fmin_l_bfgs_b(
        lambda phases: loss_and_grad_for_phases(phases, state),
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
