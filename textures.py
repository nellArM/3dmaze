"""Procedural textures generated with numpy.

Each function returns a contiguous ``uint8`` RGB array of shape ``(H, W, 3)``.
Generating textures in code keeps the project fully self-contained (no image
files to ship).  If you'd rather use real images, load them with
``pygame.image`` / Pillow and hand the bytes to ``Renderer.make_texture``.
"""

import numpy as np

_SIZE = 256  # texture resolution (square)


def _noise(shape, rng, amount):
    """Signed integer noise in [-amount, amount] for subtle surface variation."""
    return rng.integers(-amount, amount + 1, size=shape, dtype=np.int16)


def brick(seed=0):
    """Reddish brick wall: staggered rows of bricks separated by mortar."""
    rng = np.random.default_rng(seed)
    img = np.empty((_SIZE, _SIZE, 3), dtype=np.int16)

    brick_h, brick_w = 32, 64
    mortar = 5
    mortar_color = np.array([170, 165, 155], dtype=np.int16)
    brick_color = np.array([150, 60, 45], dtype=np.int16)

    for y in range(_SIZE):
        row = y // brick_h
        # Stagger every other row by half a brick.
        offset = (brick_w // 2) if (row % 2) else 0
        for x in range(_SIZE):
            xx = (x + offset) % brick_w
            yy = y % brick_h
            if yy < mortar or xx < mortar:
                img[y, x] = mortar_color
            else:
                img[y, x] = brick_color
    img += _noise(img.shape, rng, 18)
    return np.clip(img, 0, 255).astype(np.uint8)


def floor_stone(seed=1):
    """Grey stone floor with mild noise and a faint grid of seams."""
    rng = np.random.default_rng(seed)
    base = np.array([95, 95, 100], dtype=np.int16)
    img = np.empty((_SIZE, _SIZE, 3), dtype=np.int16)
    img[:] = base
    img += _noise(img.shape, rng, 22)
    tile = 64
    seam = np.array([60, 60, 65], dtype=np.int16)
    img[::tile, :] = seam
    img[:, ::tile] = seam
    return np.clip(img, 0, 255).astype(np.uint8)


def ceiling(seed=2):
    """Dark, mostly flat ceiling so it reads as overhead and not floor."""
    rng = np.random.default_rng(seed)
    base = np.array([55, 58, 68], dtype=np.int16)
    img = np.empty((_SIZE, _SIZE, 3), dtype=np.int16)
    img[:] = base
    img += _noise(img.shape, rng, 10)
    return np.clip(img, 0, 255).astype(np.uint8)


def flag(seed=4):
    """Vivid red marker texture for player-placed flags (with a lighter stripe)."""
    img = np.empty((_SIZE, _SIZE, 3), dtype=np.uint8)
    img[:] = np.array([220, 30, 30], dtype=np.uint8)
    # A lighter horizontal band so the post catches the eye through the fog.
    band = slice(_SIZE // 2 - 24, _SIZE // 2 + 24)
    img[band, :] = np.array([255, 170, 60], dtype=np.uint8)
    return img


def goal(seed=3):
    """Bright emissive texture for the exit pillar so it stands out in fog."""
    img = np.empty((_SIZE, _SIZE, 3), dtype=np.uint8)
    # Vertical green/gold bands.
    x = np.arange(_SIZE)
    band = ((x // 24) % 2).astype(bool)
    img[:, band] = np.array([80, 255, 120], dtype=np.uint8)
    img[:, ~band] = np.array([255, 230, 90], dtype=np.uint8)
    return img
