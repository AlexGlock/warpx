"""Check solver diagnostics, including aborts that CTest WILL_FAIL cannot invert."""

import subprocess
import sys

executable, case = sys.argv[1:]
result = subprocess.run(
    [executable, f"test.boundary_failure={case}"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True,
    check=False,
)
expected = {
    "eb_near_open": "the cell layer next to open faces must be free of EB",
    "periodic": "requires fixed-potential domain or embedded boundaries",
    "periodic_open": "cannot be combined with periodic",
    "all_open_1d": "Cartesian 1D open boundaries require at least one Dirichlet face",
    "invalid_reference": "larger than the domain half-diagonal",
    "unused_reference": "only used for all-open Cartesian 2D",
}[case]
assert result.returncode != 0, result.stdout
assert expected in result.stdout, result.stdout
