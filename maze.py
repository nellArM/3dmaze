"""Maze generation, BFS verification, and start/end placement.

The maze is modelled in two layers:

* A logical grid of ``n x n`` *cells* (default 100x100).
* An *occupancy grid* of ``(2*n + 1) x (2*n + 1)`` blocks where ``True`` means a
  solid wall and ``False`` means an open passage.  Cells live at the odd
  coordinates ``(2*r + 1, 2*c + 1)``; the even coordinates between two cells are
  carved open when the generator connects those cells.

This block model matches the chunky corridors of the Windows 95 "3D Maze"
screensaver and makes collision detection a simple ``grid[z][x]`` lookup.
"""

from collections import deque
import random

# Directions between *cells* in the logical grid: (d_row, d_col).
_CELL_DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def occ_dims(n):
    """Return the occupancy-grid side length for an ``n x n`` cell maze."""
    return 2 * n + 1


def cell_to_occ(r, c):
    """Convert logical cell coords to occupancy-grid coords (its passage block)."""
    return 2 * r + 1, 2 * c + 1


def generate_maze(n=100, seed=None):
    """Generate a perfect ``n x n`` maze via randomized DFS (recursive backtracker).

    Returns the occupancy grid as a list of ``2*n+1`` rows, each a list of bools
    (``True`` = wall).  A "perfect" maze has exactly one path between any two
    cells, so the result is always fully connected.
    """
    rng = random.Random(seed)
    size = occ_dims(n)
    # Start completely walled-in.
    grid = [[True] * size for _ in range(size)]

    visited = [[False] * n for _ in range(n)]
    start = (rng.randrange(n), rng.randrange(n))
    visited[start[0]][start[1]] = True
    # Carve the start cell open.
    sr, sc = cell_to_occ(*start)
    grid[sr][sc] = False

    stack = [start]
    while stack:
        r, c = stack[-1]
        # Collect unvisited neighbouring cells.
        neighbours = []
        for dr, dc in _CELL_DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < n and 0 <= nc < n and not visited[nr][nc]:
                neighbours.append((nr, nc, dr, dc))
        if not neighbours:
            stack.pop()
            continue
        nr, nc, dr, dc = rng.choice(neighbours)
        visited[nr][nc] = True
        # Carve the wall between (r,c) and (nr,nc) plus the neighbour cell itself.
        or_, oc = cell_to_occ(r, c)
        grid[or_ + dr][oc + dc] = False          # wall between the two cells
        grid[or_ + 2 * dr][oc + 2 * dc] = False  # the neighbour passage block
        stack.append((nr, nc))

    return grid


def _open_cells(grid, n):
    """Yield occupancy coords of every logical cell (all are passages)."""
    for r in range(n):
        for c in range(n):
            yield cell_to_occ(r, c)


def verify_bfs(grid, n=100):
    """BFS reachability check: confirm every one of the ``n*n`` cells is reachable.

    Returns ``True`` iff a flood fill from one cell reaches all cells.  This is
    the explicit connectivity guarantee required for the game; for a recursive
    backtracker it always passes, but we verify rather than assume.
    """
    size = occ_dims(n)
    start = cell_to_occ(0, 0)
    seen = [[False] * size for _ in range(size)]
    q = deque([start])
    seen[start[0]][start[1]] = True
    reached = 0
    while q:
        z, x = q.popleft()
        reached += 1
        for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nz, nx = z + dz, x + dx
            if 0 <= nz < size and 0 <= nx < size and not seen[nz][nx] and not grid[nz][nx]:
                seen[nz][nx] = True
                q.append((nz, nx))
    # Count how many *passage blocks* exist; all should be reached.
    total_open = sum(0 if cell else 1 for row in grid for cell in row)
    return reached == total_open


def bfs_distances(grid, start):
    """BFS step-distances over open blocks from ``start`` (a (z, x) occ coord).

    Returns a 2D list of distances; unreachable / wall blocks are ``-1``.
    """
    size = len(grid)
    dist = [[-1] * size for _ in range(size)]
    sz, sx = start
    dist[sz][sx] = 0
    q = deque([start])
    while q:
        z, x = q.popleft()
        for dz, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nz, nx = z + dz, x + dx
            if 0 <= nz < size and 0 <= nx < size and dist[nz][nx] == -1 and not grid[nz][nx]:
                dist[nz][nx] = dist[z][x] + 1
                q.append((nz, nx))
    return dist


def pick_start_and_end(grid, n=100, seed=None, far_fraction=0.25):
    """Pick a random start cell and a random end among the farthest cells.

    Returns ``(start_occ, end_occ)`` as ``(z, x)`` occupancy coordinates.  The
    end is chosen from the farthest ``far_fraction`` of reachable cells (by BFS
    distance) so the goal is never trivially next to the player.
    """
    rng = random.Random(seed)
    cells = list(_open_cells(grid, n))
    start = rng.choice(cells)
    dist = bfs_distances(grid, start)

    reachable = [(z, x) for (z, x) in cells if dist[z][x] >= 0 and (z, x) != start]
    reachable.sort(key=lambda zx: dist[zx[0]][zx[1]])
    cutoff = int(len(reachable) * (1.0 - far_fraction))
    far_cells = reachable[cutoff:] or reachable
    end = rng.choice(far_cells)
    return start, end


if __name__ == "__main__":
    # Headless sanity check.
    g = generate_maze(100)
    ok = verify_bfs(g, 100)
    print(f"BFS: all {100 * 100} cells reachable" if ok else "BFS: FAILED")
    s, e = pick_start_and_end(g, 100)
    d = bfs_distances(g, s)
    print(f"start={s} end={e} distance={d[e[0]][e[1]]} blocks")
