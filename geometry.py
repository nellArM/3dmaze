"""Build batched mesh geometry from a maze occupancy grid.

World convention: X = column, Z = row, Y = up.  Each block is a 1x1 footprint;
walls run from ``y = 0`` (floor) to ``y = WALL_H`` (ceiling).

Vertices are interleaved as ``(px, py, pz, nx, ny, nz, u, v)`` -> 8 floats.
Each quad is two triangles (6 vertices).  Only *exposed* wall faces are emitted
(a wall face only where it borders an open block), so interior block faces never
reach the GPU.  Back-face culling is left off in the renderer, so winding order
does not matter here.
"""

import numpy as np

WALL_H = 1.0  # wall / ceiling height in world units


def _quad(out, corners, normal, uvs):
    """Append a quad (as two triangles) to the flat vertex list ``out``."""
    nx, ny, nz = normal
    # Triangle fan order: 0-1-2, 0-2-3.
    for i in (0, 1, 2, 0, 2, 3):
        px, py, pz = corners[i]
        u, v = uvs[i]
        out.extend((px, py, pz, nx, ny, nz, u, v))


# Standard quad UVs (one texture repeat per block face).
_UV = ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))


def build_mesh(grid):
    """Return separate vertex arrays for ``(walls, floor, ceiling)``.

    Each is a contiguous ``float32`` array shaped ``(num_vertices, 8)`` ready for
    upload.  They are split so the renderer can bind a different texture to each.
    """
    size = len(grid)
    walls, floor, ceil = [], [], []
    h = WALL_H

    for z in range(size):
        row = grid[z]
        for x in range(size):
            if not row[x]:
                # Open block -> floor + ceiling.
                _quad(floor,
                      [(x, 0, z), (x + 1, 0, z), (x + 1, 0, z + 1), (x, 0, z + 1)],
                      (0, 1, 0), _UV)
                _quad(ceil,
                      [(x, h, z), (x + 1, h, z), (x + 1, h, z + 1), (x, h, z + 1)],
                      (0, -1, 0), _UV)
                continue

            # Solid block -> emit a vertical face toward each open neighbour.
            # East (+X)
            if x + 1 < size and not grid[z][x + 1]:
                _quad(walls,
                      [(x + 1, 0, z), (x + 1, 0, z + 1), (x + 1, h, z + 1), (x + 1, h, z)],
                      (1, 0, 0), _UV)
            # West (-X)
            if x - 1 >= 0 and not grid[z][x - 1]:
                _quad(walls,
                      [(x, 0, z + 1), (x, 0, z), (x, h, z), (x, h, z + 1)],
                      (-1, 0, 0), _UV)
            # South (+Z)
            if z + 1 < size and not grid[z + 1][x]:
                _quad(walls,
                      [(x + 1, 0, z + 1), (x, 0, z + 1), (x, h, z + 1), (x + 1, h, z + 1)],
                      (0, 0, 1), _UV)
            # North (-Z)
            if z - 1 >= 0 and not grid[z - 1][x]:
                _quad(walls,
                      [(x, 0, z), (x + 1, 0, z), (x + 1, h, z), (x, h, z)],
                      (0, 0, -1), _UV)

    def pack(data):
        arr = np.asarray(data, dtype=np.float32)
        return arr.reshape(-1, 8) if arr.size else arr.reshape(0, 8)

    return pack(walls), pack(floor), pack(ceil)


def build_pillar(end_occ, radius=0.32, height=WALL_H):
    """Build a square post mesh centred on ``end_occ``.

    ``end_occ`` is a ``(z, x)`` occupancy coordinate.  Returns a ``(N, 8)``
    float32 array (4 side faces).  Used for the tall exit pillar and, with a
    smaller radius/height, for flag markers.
    """
    z, x = end_occ
    cx, cz = x + 0.5, z + 0.5
    r = radius
    h = height
    x0, x1 = cx - r, cx + r
    z0, z1 = cz - r, cz + r
    out = []
    # Four side faces of the pillar.
    _quad(out, [(x1, 0, z0), (x1, 0, z1), (x1, h, z1), (x1, h, z0)], (1, 0, 0), _UV)
    _quad(out, [(x0, 0, z1), (x0, 0, z0), (x0, h, z0), (x0, h, z1)], (-1, 0, 0), _UV)
    _quad(out, [(x1, 0, z1), (x0, 0, z1), (x0, h, z1), (x1, h, z1)], (0, 0, 1), _UV)
    _quad(out, [(x0, 0, z0), (x1, 0, z0), (x1, h, z0), (x0, h, z0)], (0, 0, -1), _UV)
    return np.asarray(out, dtype=np.float32).reshape(-1, 8)


def build_flags(cells, radius=0.12, height=0.7):
    """Build a batched mesh of thin flag posts, one per cell in ``cells``.

    ``cells`` is an iterable of ``(z, x)`` occupancy coordinates.  Returns a
    ``(N, 8)`` float32 array; ``(0, 8)`` when there are no flags.
    """
    posts = [build_pillar(c, radius=radius, height=height) for c in cells]
    if not posts:
        return np.zeros((0, 8), dtype=np.float32)
    return np.concatenate(posts, axis=0)
