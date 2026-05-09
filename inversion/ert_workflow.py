"""End-to-end 3D ERT simulation and inversion workflow in pyGIMLi."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pygimli as pg
import pygimli.physics.ert as ert


@dataclass(frozen=True)
class SurveyConfig:
    """L-array style 3D survey configuration."""

    n_electrodes_leg: int = 24
    spacing: float = 2.5
    origin: Tuple[float, float, float] = (-30.0, -10.0, 0.0)
    noise_level: float = 0.02
    noise_abs: float = 1e-5


def build_l_array(config: SurveyConfig = SurveyConfig()) -> pg.PosVector:
    """Create an L-array electrode geometry (two orthogonal survey lines)."""
    ox, oy, oz = config.origin

    x_line = [[ox + i * config.spacing, oy, oz] for i in range(config.n_electrodes_leg)]
    y_line = [[ox, oy + i * config.spacing, oz] for i in range(1, config.n_electrodes_leg)]

    sensors = pg.PosVector()
    for p in (x_line + y_line):
        sensors.push_back(pg.Pos(*p))
    return sensors


def create_measurement_scheme(sensors: pg.PosVector) -> pg.DataContainerERT:
    """Create a dipole-dipole measurement scheme over the L-array sensors."""
    return ert.createData(sensors=sensors, schemeName="dd")


def build_resistivity_model(mesh: pg.Mesh,
                            background_rho: float = 100.0,
                            karst_rho: float = 30.0) -> pg.Vector:
    """Assign resistivity values to regions, with low-resistivity karst anomaly."""
    rhomap = [[1, background_rho], [2, karst_rho]]
    return ert.createModel(mesh=mesh, rhomap=rhomap)


def simulate_ert_data(mesh: pg.Mesh,
                      scheme: pg.DataContainerERT,
                      res_model: pg.Vector,
                      config: SurveyConfig = SurveyConfig()) -> pg.DataContainerERT:
    """Forward-simulate noisy apparent resistivity data."""
    return ert.simulate(
        mesh=mesh,
        scheme=scheme,
        res=res_model,
        noiseLevel=config.noise_level,
        noiseAbs=config.noise_abs,
        seed=42,
        sr=True,
        calcOnly=False,
    )


def run_inversion(data: pg.DataContainerERT,
                  lam: float = 20.0,
                  max_iter: int = 10) -> tuple[ert.ERTManager, pg.Vector]:
    """Run ERT inversion and return manager plus inversion model."""
    manager = ert.ERTManager(data)
    model = manager.invert(lam=lam, maxIter=max_iter, verbose=True)
    return manager, model
