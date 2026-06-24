"""Flag (breadcrumb) state — small, GL-free, and unit-testable.

The player can drop up to ``MAX`` flags on grid cells and pick them back up.
``cell`` values are ``(z, x)`` occupancy coordinates, matching ``Player.cell``.
"""


class FlagSet:
    MAX = 3

    def __init__(self):
        self.cells = []  # placed flags, in placement order

    @property
    def remaining(self):
        """How many flags the player can still place."""
        return self.MAX - len(self.cells)

    def toggle(self, cell):
        """Drop or pick up a flag at ``cell``.

        Returns one of:
          * ``"removed"`` — ``cell`` was flagged, now picked up.
          * ``"placed"``  — ``cell`` was empty and a flag was dropped.
          * ``"full"``    — ``cell`` was empty but all flags are already in use.
        """
        if cell in self.cells:
            self.cells.remove(cell)
            return "removed"
        if len(self.cells) >= self.MAX:
            return "full"
        self.cells.append(cell)
        return "placed"

    def clear(self):
        """Remove all flags (used when a new maze is generated)."""
        self.cells.clear()
