"""Nonlinear coupled-tank digital twin and PI control evaluation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class TankParameters:
    area1: float = 0.015
    area2: float = 0.015
    outlet1: float = 4.0e-5
    outlet2: float = 4.2e-5
    coupling: float = 2.8e-5
    pump_gain1: float = 1.55e-4
    pump_gain2: float = 1.50e-4
    # A slow pump/valve train creates the real tuning trade-off: integral
    # action improves rise time but stored actuator flow can cause overflow.
    pump_time_constant: float = 3.5
    gravity: float = 9.81
    safe_min: float = 0.055
    # The main operating point is 0.35 m; 0.36 m is the process high-high
    # alarm, leaving only 10 mm of headroom during online tuning.
    safe_max: float = 0.36
    physical_max: float = 0.55
    max_saturation_fraction: float = 0.50


@dataclass(frozen=True)
class SimulationConfig:
    duration: float = 110.0
    dt: float = 0.1
    initial_height: tuple[float, float] = (0.14, 0.12)
    measurement_noise: float = 7.5e-4
    parameter_variation: float = 0.02


@dataclass
class SimulationResult:
    time: NDArray[np.float64]
    height: NDArray[np.float64]
    measured_height: NDArray[np.float64]
    reference: NDArray[np.float64]
    command: NDArray[np.float64]
    pump_state: NDArray[np.float64]
    cost: float
    iae: float
    energy: float
    total_variation: float
    safety_margin: float
    high_margin: float
    low_margin: float
    actuator_margin: float
    saturation_fraction: float
    violation_steps: int

    @property
    def safe(self) -> bool:
        return self.safety_margin >= 0.0


@dataclass(frozen=True)
class Evaluation:
    objective: float
    safety: float
    cost: float
    safe: bool
    violation_steps: int


class PIController:
    """Two decentralized PI loops with feedforward and anti-windup."""

    def __init__(self, gains: ArrayLike, dt: float) -> None:
        gains_array = np.asarray(gains, dtype=float)
        if gains_array.shape != (4,) or np.any(gains_array <= 0.0):
            raise ValueError("gains must be [Kp1, Ki1, Kp2, Ki2] and positive")
        self.kp = gains_array[[0, 2]]
        self.ki = gains_array[[1, 3]]
        self.dt = dt
        self.integral = np.zeros(2, dtype=float)

    def update(
        self,
        reference: NDArray[np.float64],
        measurement: NDArray[np.float64],
        feedforward: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        error = reference - measurement
        candidate_integral = self.integral + error * self.dt
        raw = feedforward + self.kp * error + self.ki * candidate_integral
        command = np.clip(raw, 0.0, 1.0)

        # Integrate unless saturation would be driven further into its limit.
        blocked_high = (raw > 1.0) & (error > 0.0)
        blocked_low = (raw < 0.0) & (error < 0.0)
        update_mask = ~(blocked_high | blocked_low)
        self.integral[update_mask] = candidate_integral[update_mask]
        return command


class CoupledTankPlant:
    """Software-in-the-loop nonlinear plant used as an expensive black box."""

    def __init__(
        self,
        parameters: TankParameters | None = None,
        config: SimulationConfig | None = None,
    ) -> None:
        self.parameters = parameters or TankParameters()
        self.config = config or SimulationConfig()

    def reference(self, time: float) -> NDArray[np.float64]:
        if time < 10.0:
            return np.array([0.18, 0.15])
        if time < 62.0:
            return np.array([0.35, 0.285])
        return np.array([0.255, 0.335])

    def _feedforward(
        self,
        reference: NDArray[np.float64],
        pump_gains: NDArray[np.float64],
        outlet_areas: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        p = self.parameters
        delta = reference[0] - reference[1]
        cross = p.coupling * np.sign(delta) * np.sqrt(2.0 * p.gravity * abs(delta))
        outlets = outlet_areas * np.sqrt(2.0 * p.gravity * np.maximum(reference, 0.0))
        required = np.array([outlets[0] + cross, outlets[1] - cross])
        return np.clip(required / pump_gains, 0.0, 1.0)

    def simulate(self, gains: ArrayLike, seed: int = 0) -> SimulationResult:
        p = self.parameters
        c = self.config
        rng = np.random.default_rng(seed)
        steps = int(round(c.duration / c.dt)) + 1
        time = np.linspace(0.0, c.duration, steps)
        height = np.empty((steps, 2), dtype=float)
        measured = np.empty_like(height)
        reference = np.empty_like(height)
        command = np.empty_like(height)
        pump_state = np.empty_like(height)

        height[0] = np.asarray(c.initial_height, dtype=float)
        pump_state[0] = 0.0
        reference[0] = self.reference(time[0])
        measured[0] = height[0] + rng.normal(0.0, c.measurement_noise, 2)

        # Each rollout has small unknown plant mismatch and one random leak.
        pump_gains = np.array([p.pump_gain1, p.pump_gain2]) * rng.normal(
            1.0, c.parameter_variation, 2
        )
        outlet_areas = np.array([p.outlet1, p.outlet2]) * rng.normal(
            1.0, c.parameter_variation, 2
        )
        leak_start = rng.uniform(72.0, 82.0)
        leak_duration = rng.uniform(7.0, 11.0)
        leak_rate = rng.uniform(0.6e-5, 1.4e-5)

        controller = PIController(gains, c.dt)
        command[0] = controller.update(
            reference[0],
            measured[0],
            self._feedforward(reference[0], pump_gains, outlet_areas),
        )

        areas = np.array([p.area1, p.area2])
        for k in range(steps - 1):
            pump_state[k + 1] = pump_state[k] + c.dt * (
                command[k] - pump_state[k]
            ) / p.pump_time_constant
            pump_state[k + 1] = np.clip(pump_state[k + 1], 0.0, 1.0)

            h = np.maximum(height[k], 0.0)
            delta = h[0] - h[1]
            cross = p.coupling * np.sign(delta) * np.sqrt(
                2.0 * p.gravity * abs(delta)
            )
            outflow = outlet_areas * np.sqrt(2.0 * p.gravity * h)
            inflow = pump_gains * pump_state[k + 1]
            leak = leak_rate if leak_start <= time[k] <= leak_start + leak_duration else 0.0
            derivative = np.array(
                [inflow[0] - outflow[0] - cross - leak, inflow[1] - outflow[1] + cross]
            ) / areas
            height[k + 1] = np.clip(height[k] + c.dt * derivative, 0.0, p.physical_max)

            reference[k + 1] = self.reference(time[k + 1])
            measured[k + 1] = height[k + 1] + rng.normal(
                0.0, c.measurement_noise, 2
            )
            command[k + 1] = controller.update(
                reference[k + 1],
                measured[k + 1],
                self._feedforward(reference[k + 1], pump_gains, outlet_areas),
            )

        error = reference - height
        iae = float(np.trapezoid(np.mean(np.abs(error), axis=1), time))
        energy = float(np.trapezoid(np.mean(command**2, axis=1), time))
        total_variation = float(np.sum(np.mean(np.abs(np.diff(command, axis=0)), axis=1)))

        normalized_iae = iae / (c.duration * 0.20)
        normalized_energy = energy / c.duration
        normalized_tv = total_variation / 15.0
        cost = 0.72 * normalized_iae + 0.20 * normalized_energy + 0.08 * normalized_tv

        high_margin_m = p.safe_max - float(np.max(height))
        low_margin_m = float(np.min(height)) - p.safe_min
        margin_scale = 0.02
        high_margin = high_margin_m / margin_scale
        low_margin = low_margin_m / margin_scale
        level_violations = (height > p.safe_max) | (height < p.safe_min)
        saturated = (command >= 0.995) | (command <= 0.005)
        saturation_fraction = float(np.mean(saturated))
        actuator_margin = (p.max_saturation_fraction - saturation_fraction) / 0.10
        safety_margin = min(high_margin, low_margin, actuator_margin)
        excess_saturation_steps = max(
            0,
            int(np.count_nonzero(saturated) - p.max_saturation_fraction * saturated.size),
        )

        return SimulationResult(
            time=time,
            height=height,
            measured_height=measured,
            reference=reference,
            command=command,
            pump_state=pump_state,
            cost=float(cost),
            iae=iae,
            energy=energy,
            total_variation=total_variation,
            safety_margin=float(safety_margin),
            high_margin=float(high_margin),
            low_margin=float(low_margin),
            actuator_margin=float(actuator_margin),
            saturation_fraction=saturation_fraction,
            violation_steps=int(np.count_nonzero(level_violations)) + excess_saturation_steps,
        )

    def evaluate(self, gains: ArrayLike, seed: int = 0) -> Evaluation:
        result = self.simulate(gains, seed=seed)
        return Evaluation(
            objective=-result.cost,
            safety=result.safety_margin,
            cost=result.cost,
            safe=result.safe,
            violation_steps=result.violation_steps,
        )
