"""First-person player: camera state, smooth WASD movement, wall collision.

The player is a circle of radius ``RADIUS`` on the XZ plane at a fixed eye
height.  Movement resolves the X and Z axes independently so the player slides
along walls instead of sticking to them.

Controls (tank style, per the game spec):
    W -> forward      S -> backward
    A -> rotate left  D -> rotate right
"""

import math

RADIUS = 0.25       # collision radius in world units (block = 1.0)
EYE_HEIGHT = 0.5    # camera height between floor (0) and ceiling (WALL_H = 1)
MOVE_SPEED = 3.5    # blocks per second
TURN_SPEED = 2.6    # radians per second


class Player:
    def __init__(self, start_occ, yaw=0.0):
        # start_occ is a (z, x) occupancy coord; centre the player in that block.
        z, x = start_occ
        self.x = x + 0.5
        self.z = z + 0.5
        self.y = EYE_HEIGHT
        self.yaw = yaw  # radians; 0 looks toward +Z

    @property
    def cell(self):
        """Current occupancy block coordinate as (z, x) ints."""
        return int(self.z), int(self.x)

    def forward(self):
        """Unit facing vector on the XZ plane."""
        return math.sin(self.yaw), math.cos(self.yaw)

    def update(self, dt, keys, grid):
        """Advance one frame. ``keys`` is a dict-like of bools by intent name."""
        # Rotation. The camera faces +Z (see renderer.look_at), which mirrors
        # the handedness: turning the view left requires *increasing* yaw and
        # turning right requires *decreasing* it.
        if keys.get("left"):
            self.yaw += TURN_SPEED * dt
        if keys.get("right"):
            self.yaw -= TURN_SPEED * dt

        # Forward / backward intent along the facing vector.
        move = 0.0
        if keys.get("forward"):
            move += 1.0
        if keys.get("back"):
            move -= 1.0
        if move == 0.0:
            return

        fx, fz = self.forward()
        step = move * MOVE_SPEED * dt
        dx = fx * step
        dz = fz * step

        # Resolve each axis separately so we slide along walls.
        if not self._blocked(self.x + math.copysign(RADIUS, dx) + dx, self.z, grid):
            self.x += dx
        if not self._blocked(self.x, self.z + math.copysign(RADIUS, dz) + dz, grid):
            self.z += dz

    def _blocked(self, x, z, grid):
        """Is world point (x, z) inside a solid block (or out of bounds)?"""
        gx, gz = int(x), int(z)
        if gz < 0 or gx < 0 or gz >= len(grid) or gx >= len(grid[0]):
            return True
        return grid[gz][gx]
