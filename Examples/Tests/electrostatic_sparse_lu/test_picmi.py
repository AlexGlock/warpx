"""Check the WarpX method extension without loading a compiled Python library."""

from pydantic import ValidationError

import picmistandard
import pywarpx
from pywarpx import picmi


def grid(dim, **kwargs):
    cls = {
        1: picmi.Cartesian1DGrid,
        2: picmi.Cartesian2DGrid,
        3: picmi.Cartesian3DGrid,
    }[dim]
    inputs = dict(
        number_of_cells=[16] * dim,
        lower_bound=[0.0] * dim,
        upper_bound=[1.0] * dim,
        lower_boundary_conditions=["dirichlet"] * dim,
        upper_boundary_conditions=["dirichlet"] * dim,
    )
    inputs.update(kwargs)
    return cls(**inputs)


def expect_invalid(message, **kwargs):
    try:
        picmi.ElectrostaticSolver(**kwargs)
    except ValidationError as exc:
        assert message in str(exc), str(exc)
    else:
        raise AssertionError(f"Expected validation error: {message}")


original_methods = list(picmistandard.PICMI_ElectrostaticSolver.methods_list)
original_method_type = picmistandard.PICMI_ElectrostaticSolver.model_fields[
    "method"
].annotation
for dim in (1, 2, 3):
    for method in ("sparse_lu", "SPARSE_LU"):
        solver = picmi.ElectrostaticSolver(
            grid=grid(dim), method=method, warpx_self_fields_verbosity=0
        )
        assert solver.method == "sparse_lu"
        # Exercise Pydantic deserialization as well as direct construction.
        solver = picmi.ElectrostaticSolver.model_validate(solver.model_dump())
        solver.solver_initialize_inputs()
        assert pywarpx.warpx.poisson_solver == "sparse_lu"
        assert pywarpx.warpx.do_electrostatic == "labframe"
        assert pywarpx.warpx.self_fields_verbosity == 0
        assert pywarpx.amr.max_level == 0

assert picmistandard.PICMI_ElectrostaticSolver.methods_list == original_methods
assert (
    picmistandard.PICMI_ElectrostaticSolver.model_fields["method"].annotation
    == original_method_type
)
for method, expected in (
    (None, None),
    ("Multigrid", "Multigrid"),
    ("multigrid", "Multigrid"),
    ("FFT", "FFT"),
    ("fft", "FFT"),
):
    assert picmi.ElectrostaticSolver(grid=grid(3), method=method).method == expected

expect_invalid("method", grid=grid(2), method="unknown_solver")
solver = picmi.ElectrostaticSolver(grid=grid(2), method="Multigrid")
solver.method = "SPARSE_LU"
assert solver.method == "sparse_lu"

for kw in (
    {"warpx_relativistic": True},
    {"warpx_magnetostatic": True},
    {"warpx_effective_potential": True},
):
    expect_invalid("labframe", grid=grid(2), method="sparse_lu", **kw)

expect_invalid(
    "mesh refinement",
    grid=grid(2, refined_regions=[[1, [0.25, 0.25], [0.75, 0.75]]]),
    method="sparse_lu",
)
expect_invalid(
    "Cartesian 1D, 2D and 3D",
    grid=picmi.CylindricalGrid(
        number_of_cells=[16, 16],
        lower_bound=[0.0, 0.0],
        upper_bound=[1.0, 1.0],
        lower_boundary_conditions=[None, "dirichlet"],
        upper_boundary_conditions=["dirichlet", "dirichlet"],
    ),
    method="sparse_lu",
)
for side in ("lower_boundary_conditions", "upper_boundary_conditions"):
    expect_invalid(
        "Dirichlet",
        grid=grid(2, **{side: ["neumann", "dirichlet"]}),
        method="sparse_lu",
    )

for dim in (2, 3):
    for bc in ("periodic", "open"):
        boundaries = [bc] * (dim - 1) + ["dirichlet"]
        solver = picmi.ElectrostaticSolver(
            grid=grid(
                dim,
                lower_boundary_conditions=boundaries,
                upper_boundary_conditions=boundaries,
            ),
            method="sparse_lu",
        )
        solver.solver_initialize_inputs()
        assert pywarpx.boundary.field_lo == [bc] * (dim - 1) + ["pec"]
        assert pywarpx.boundary.field_hi == [bc] * (dim - 1) + ["pec"]

for lower, upper in (
    ("dirichlet", "dirichlet"),
    ("dirichlet", "open"),
    ("open", "dirichlet"),
):
    solver = picmi.ElectrostaticSolver(
        grid=grid(
            1,
            lower_boundary_conditions=[lower],
            upper_boundary_conditions=[upper],
        ),
        method="sparse_lu",
    )
    solver.solver_initialize_inputs()
    assert pywarpx.boundary.field_lo == ["pec" if lower == "dirichlet" else "open"]
    assert pywarpx.boundary.field_hi == ["pec" if upper == "dirichlet" else "open"]

for boundary in ("open", "periodic"):
    expect_invalid(
        "at least one Dirichlet boundary",
        grid=grid(
            1,
            lower_boundary_conditions=[boundary],
            upper_boundary_conditions=[boundary],
        ),
        method="sparse_lu",
    )

# Fully periodic domains may have embedded conductors, which are supplied by
# Simulation. The C++ operator must check anchoring after the EB is constructed.
picmi.ElectrostaticSolver(
    grid=grid(
        2,
        lower_boundary_conditions=["periodic"] * 2,
        upper_boundary_conditions=["periodic"] * 2,
    ),
    method="sparse_lu",
)
expect_invalid(
    "paired periodic",
    grid=grid(2, lower_boundary_conditions=["periodic", "dirichlet"]),
    method="sparse_lu",
)
expect_invalid(
    "cannot be combined with periodic",
    grid=grid(
        2,
        lower_boundary_conditions=["periodic", "open"],
        upper_boundary_conditions=["periodic", "open"],
    ),
    method="sparse_lu",
)
for dim in (2, 3):
    for wall in range(-1, 2 * dim):
        lower, upper = ["open"] * dim, ["open"] * dim
        if wall >= 0:
            (lower if wall % 2 == 0 else upper)[wall // 2] = "dirichlet"
        solver = picmi.ElectrostaticSolver(
            grid=grid(
                dim,
                number_of_cells=[1] * dim,
                lower_boundary_conditions=lower,
                upper_boundary_conditions=upper,
            ),
            method="sparse_lu",
        )
        solver.solver_initialize_inputs()
        expected = 3 * dim**0.5 if dim == 2 and wall == -1 else None
        assert pywarpx.warpx.sparse_lu_open_reference_radius == expected

open_grid = grid(
    2,
    lower_bound=[-2.0, 3.0],
    upper_bound=[1.0, 7.0],
    lower_boundary_conditions=["open"] * 2,
    upper_boundary_conditions=["open"] * 2,
)
for radius, expected in ((None, 15.0), (10.0, 10.0)):
    solver = picmi.ElectrostaticSolver(
        grid=open_grid,
        method="sparse_lu",
        warpx_sparse_lu_open_reference_radius=radius,
    )
    solver = picmi.ElectrostaticSolver.model_validate(solver.model_dump())
    solver.solver_initialize_inputs()
    assert pywarpx.warpx.sparse_lu_open_reference_radius == expected
for radius in (0.1, 2.5):
    expect_invalid(
        "half-diagonal",
        grid=open_grid,
        method="sparse_lu",
        warpx_sparse_lu_open_reference_radius=radius,
    )
for radius in (0, -1, float("nan"), float("inf")):
    expect_invalid(
        "sparse_lu_open_reference_radius",
        grid=open_grid,
        method="sparse_lu",
        warpx_sparse_lu_open_reference_radius=radius,
    )
for other_grid in (
    grid(1),
    grid(2),
    grid(
        3,
        lower_boundary_conditions=["open"] * 3,
        upper_boundary_conditions=["open"] * 3,
    ),
):
    expect_invalid(
        "only used for all-open 2D",
        grid=other_grid,
        method="sparse_lu",
        warpx_sparse_lu_open_reference_radius=10.0,
    )
expect_invalid(
    "requires method='sparse_lu'",
    grid=open_grid,
    method="Multigrid",
    warpx_sparse_lu_open_reference_radius=10.0,
)
# A subsequent non-open solver must clear the previous solver's radius input.
picmi.ElectrostaticSolver(grid=grid(2)).solver_initialize_inputs()
assert pywarpx.warpx.sparse_lu_open_reference_radius is None
