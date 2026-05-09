"""3D mesh construction utilities for pyGIMLi ERT workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import pygimli as pg
import pygimli.meshtools as mt


@dataclass(frozen=True)
class MeshConfig:
    """Domain and discretization configuration for a 3D ERT model volume."""

    x_range: Tuple[float, float] = (-40.0, 40.0)
    y_range: Tuple[float, float] = (-20.0, 20.0)
    z_range: Tuple[float, float] = (-30.0, 0.0)
    quality: float = 1.3
    area: float = 2.0


def create_world(config: MeshConfig) -> pg.Mesh:
    """Create a 3D world PLC for meshing."""
    start = [config.x_range[0], config.y_range[0], config.z_range[0]]
    end = [config.x_range[1], config.y_range[1], config.z_range[1]]
    world = mt.createCube(start=start, end=end, marker=1, boundaryMarker=1)
    return world


def add_karst_anomaly(world: pg.Mesh,
                     center: Tuple[float, float, float] = (0.0, 0.0, -12.0),
                     radius: float = 5.0,
                     marker: int = 2) -> pg.Mesh:
    """Add a spherical karst anomaly to the PLC using a unique region marker."""
    karst = mt.createSphere(pos=list(center), radius=radius, marker=marker, boundaryMarker=2)
    return world + karst


def create_3d_mesh(config: MeshConfig = MeshConfig()) -> pg.Mesh:
    """Build the full 3D mesh with an embedded karst anomaly."""
    world = create_world(config)
    geom = add_karst_anomaly(world)
    mesh = mt.createMesh(geom, quality=config.quality, area=config.area)
    return mesh
