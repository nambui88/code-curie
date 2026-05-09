"""Script entrypoint for 3D ERT workflow with pyGIMLi."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pygimli as pg

from inversion.ert_workflow import (
    build_l_array,
    build_resistivity_model,
    create_measurement_scheme,
    run_inversion,
    simulate_ert_data,
)
from mesh.build_mesh import MeshConfig, create_3d_mesh


def plot_results(mesh: pg.Mesh, true_model: pg.Vector, inv_model: pg.Vector, output_dir: Path) -> None:
    """Render slices of true and inverted models."""
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    pg.show(mesh, data=true_model, ax=axes[0], label="True resistivity (ohm.m)", cMap="viridis")
    axes[0].set_title("True 3D model")

    pg.show(mesh, data=inv_model, ax=axes[1], label="Inverted resistivity (ohm.m)", cMap="viridis")
    axes[1].set_title("Inversion result")

    fig.savefig(output_dir / "ert_3d_result.png", dpi=200)
    plt.close(fig)


def main() -> None:
    mesh = create_3d_mesh(MeshConfig())
    sensors = build_l_array()
    scheme = create_measurement_scheme(sensors)

    res_model = build_resistivity_model(mesh)
    data = simulate_ert_data(mesh, scheme, res_model)
    manager, inv_model = run_inversion(data)

    out = Path("outputs")
    out.mkdir(exist_ok=True)
    mesh.save(str(out / "mesh_3d.bms"))
    data.save(str(out / "simulated_data.ohm"))
    manager.paraDomain.save(str(out / "inversion_domain.bms"))
    pg.Vector(inv_model).save(str(out / "inversion_model.vector"))

    plot_results(mesh, res_model, inv_model, out)
    print(f"Saved outputs to: {out.resolve()}")


if __name__ == "__main__":
    main()
