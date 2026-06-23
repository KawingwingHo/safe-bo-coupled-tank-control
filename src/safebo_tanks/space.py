"""Controller parameter search space."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray


@dataclass(frozen=True)
class ControllerSpace:
    """Log-scaled search space for two independent PI loops.

    The optimizer works on a normalized unit hypercube. Log scaling prevents
    the Gaussian process from spending most of its resolution on large gains.
    """

    lower: tuple[float, ...] = (0.4, 0.003, 0.4, 0.003)
    upper: tuple[float, ...] = (30.0, 1.5, 30.0, 1.5)
    names: tuple[str, ...] = ("Kp1", "Ki1", "Kp2", "Ki2")

    @property
    def dimension(self) -> int:
        return len(self.lower)

    def to_gains(self, x: ArrayLike) -> NDArray[np.float64]:
        normalized = np.asarray(x, dtype=float)
        if normalized.shape[-1] != self.dimension:
            raise ValueError(f"Expected last dimension {self.dimension}")
        if np.any((normalized < 0.0) | (normalized > 1.0)):
            raise ValueError("Normalized parameters must lie in [0, 1]")
        low = np.log(np.asarray(self.lower))
        high = np.log(np.asarray(self.upper))
        return np.exp(low + normalized * (high - low))

    def normalize(self, gains: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(gains, dtype=float)
        if values.shape[-1] != self.dimension:
            raise ValueError(f"Expected last dimension {self.dimension}")
        if np.any(values <= 0.0):
            raise ValueError("PI gains must be positive")
        low = np.log(np.asarray(self.lower))
        high = np.log(np.asarray(self.upper))
        normalized = (np.log(values) - low) / (high - low)
        if np.any((normalized < -1e-12) | (normalized > 1.0 + 1e-12)):
            raise ValueError("Gains fall outside the configured search space")
        return np.clip(normalized, 0.0, 1.0)

    def conservative_seed(self) -> NDArray[np.float64]:
        return self.normalize((2.0, 0.03, 2.0, 0.03))

    def certified_initial_design(self) -> NDArray[np.float64]:
        """Small engineering-certified neighborhood around the seed.

        These are not claimed safe by the optimizer. They represent low-gain
        commissioning points checked before online optimization begins.
        """

        seed = self.conservative_seed()
        offsets = np.array(
            [
                [0.0, 0.0, 0.0, 0.0],
                [0.035, 0.0, 0.0, 0.0],
                [0.0, 0.035, 0.0, 0.0],
                [0.0, 0.0, 0.035, 0.0],
                [0.0, 0.0, 0.0, 0.035],
            ]
        )
        return np.clip(seed + offsets, 0.0, 1.0)
