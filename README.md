# 3D Maze

A first-person 3D maze game inspired by the classic Windows 95 "3D Maze"
screensaver. You spawn inside a randomly generated 100×100 maze and must walk
through the textured corridors until you reach the glowing exit pillar.

## Requirements

- Python 3.10+ (tested on 3.12)
- A GPU/driver supporting OpenGL 3.3+
- Windows, macOS, or Linux

## Install

```bash
python -m pip install pygame moderngl numpy
```

(Optionally create a virtual environment first: `python -m venv venv` then activate it.)

## Run

```bash
python main.py
```

A window opens showing the first-person view from inside the maze. Each launch
(and each press of `R`) generates a brand-new random maze.

## How to play

Find the glowing exit pillar somewhere in the maze and walk into it to win.

### Controls

| Key     | Action                          |
|---------|---------------------------------|
| `W`     | Move forward                    |
| `S`     | Move backward                   |
| `A`     | Rotate left                     |
| `D`     | Rotate right                    |
| `Space` | Drop / pick up a flag           |
| `R`     | New random maze                 |
| `Esc`   | Quit                            |

## Rules

- The maze is a randomly generated 100×100 grid. Walls are solid — you cannot
  walk through them.
- Every maze is guaranteed solvable: after generation a BFS reachability check
  confirms every cell is reachable, and the exit is placed far from your start.
- You and the exit are placed at random locations each game.
- Reach the exit pillar to win. A "You reached the exit!" message appears; press
  `R` to play a fresh maze.

### Flags

- Press `Space` to drop a red flag on the cell you're standing on — handy
  breadcrumbs for marking where you've been.
- Press `Space` again while standing on a flagged cell to pick the flag back up.
- You have **3 flags** total. The bottom-right HUD (`Flags: N/3`) shows how many
  you can still place. Flags reset when you start a new maze.

## How it works

- **Maze generation:** randomized depth-first search (recursive backtracker)
  produces a perfect maze, expanded to a block grid for rendering and collision.
- **Verification:** breadth-first search confirms full connectivity before play.
- **Rendering:** OpenGL via `moderngl` (batched geometry, procedural
  brick/stone textures, per-pixel lighting and distance fog). Window, input, and
  timing via `pygame`.

## Tests

The non-rendering logic (maze generation, BFS, geometry, movement/collision,
matrix math, textures) is covered by a `unittest` suite that needs no extra
dependencies:

```bash
python -m unittest discover -s tests -v
```

51 tests cover, among other things: maze dimensions and solid borders, the
"perfect maze" open-block invariant (`2*n*n - 1`), BFS full reachability (and
detection of a deliberately disconnected maze), exit placement in the far
region, exposed-face mesh counts, unit axis-aligned normals, walking blocked by
walls, sliding along walls, A/D turning the view the correct way (left/right),
flag placement/pickup rules (max 3, toggle, counter), and view/projection
matrix correctness.

## Project layout

| File          | Responsibility                                            |
|---------------|-----------------------------------------------------------|
| `maze.py`     | Maze generation, BFS verification, start/exit placement   |
| `textures.py` | Procedural brick / stone / ceiling / goal textures        |
| `geometry.py` | Batched mesh from the maze grid + exit pillar             |
| `player.py`   | Camera, smooth WASD movement, wall-sliding collision      |
| `flags.py`    | Flag placement/pickup state (max 3)                       |
| `renderer.py` | moderngl shaders, textures, matrices, draw loop           |
| `main.py`     | Window, game loop, input, HUD, win detection              |
