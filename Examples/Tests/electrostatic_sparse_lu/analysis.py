"""Equal nonzero conductor potentials must produce zero electric field."""

import math
from pathlib import Path

rows = [
    [float(value) for value in line.split()]
    for line in Path("diags/reducedfiles/field_max.txt").read_text().splitlines()
    if line and not line.startswith("#")
]
assert len(rows) == 3
for row in rows:
    # Ex, Ey, Ez, |E|, Bx, By, Bz, |B|. Allow single-precision roundoff.
    assert all(math.isfinite(value) and abs(value) < 1.0e-3 for value in row[2:])
