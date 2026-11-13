"""Particle start positions shared by the synthetic and MOI benchmark cases."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ParticleSeeds:
    lon_deg: np.ndarray
    lat_deg: np.ndarray
    depth_m: np.ndarray
