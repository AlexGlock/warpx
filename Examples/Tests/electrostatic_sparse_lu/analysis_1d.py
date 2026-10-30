"""Check the Cartesian 1D sparse-LU solve through WarpX's timestep path."""

import math
from pathlib import Path

field_rows = [
    [float(value) for value in line.split()]
    for line in Path("diags/reducedfiles/field_max.txt").read_text().splitlines()
    if line and not line.startswith("#")
]
assert len(field_rows) == 3
for row in field_rows:
    assert all(math.isfinite(value) and abs(value) < 1.0e-3 for value in row[2:])
