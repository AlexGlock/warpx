"""Check the WarpX method extension without loading a compiled Python library."""

import picmistandard
import pywarpx
from pywarpx import picmi


def grid(dim, **kwargs):
    cls = picmi.Cartesian2DGrid if dim == 2 else picmi.Cartesian3DGrid
    return cls(
        number_of_cells=[16] * dim,
        lower_bound=[0.0] * dim,
        upper_bound=[1.0] * dim,
        lower_boundary_conditions=["dirichlet"] * dim,
        upper_boundary_conditions=["dirichlet"] * dim,
        **kwargs,
    )


original_methods = list(picmistandard.PICMI_ElectrostaticSolver.methods_list)
for dim in (2, 3):
    solver = picmi.ElectrostaticSolver(grid=grid(dim), method="sparse_lu")
    solver.solver_initialize_inputs()
    assert pywarpx.warpx.poisson_solver == "sparse_lu"
    assert pywarpx.warpx.do_electrostatic == "labframe"
    assert pywarpx.amr.max_level == 0

assert picmistandard.PICMI_ElectrostaticSolver.methods_list == original_methods
for method in (None, "Multigrid", "FFT"):
    assert picmi.ElectrostaticSolver(grid=grid(2), method=method).method == method

for kw in (
    {"warpx_relativistic": True},
    {"warpx_magnetostatic": True},
    {"warpx_effective_potential": True},
):
    try:
        picmi.ElectrostaticSolver(grid=grid(2), method="sparse_lu", **kw)
    except ValueError:
        pass
    else:
        raise AssertionError("Unsupported sparse LU mode was accepted")

try:
    picmi.ElectrostaticSolver(
        grid=grid(2, refined_regions=[[1, [0.25, 0.25], [0.75, 0.75]]]),
        method="sparse_lu",
    )
except ValueError:
    pass
else:
    raise AssertionError("Sparse LU accepted mesh refinement")
