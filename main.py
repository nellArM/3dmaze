"""3D Maze — first-person maze game (pygame + moderngl).

Run with:  python main.py
Controls:  W/S move, A/D rotate, R new maze, Esc quit.
"""

import sys
import numpy as np
import pygame
import moderngl

import maze
import textures
import geometry
from player import Player
from renderer import Renderer
from flags import FlagSet

MAZE_N = 100
WINDOW_SIZE = (1024, 768)


# --- 2D text overlay drawn on top of the GL scene ---------------------------
OVERLAY_VERT = """
#version 330
in vec2 in_pos;
in vec2 in_uv;
out vec2 v_uv;
void main() {
    v_uv = in_uv;
    gl_Position = vec4(in_pos, 0.0, 1.0);
}
"""
OVERLAY_FRAG = """
#version 330
in vec2 v_uv;
uniform sampler2D tex;
out vec4 frag;
void main() { frag = texture(tex, v_uv); }
"""


class Overlay:
    def __init__(self, ctx):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=OVERLAY_VERT, fragment_shader=OVERLAY_FRAG)
        self.font = pygame.font.SysFont("consolas", 24)
        self.big = pygame.font.SysFont("consolas", 56, bold=True)

    def draw_text(self, text, px, py, screen, color=(235, 235, 245), font=None):
        font = font or self.font
        surf = font.render(text, True, color).convert_alpha()
        w, h = surf.get_size()
        data = pygame.image.tostring(surf, "RGBA", True)
        tex = self.ctx.texture((w, h), 4, data)
        tex.filter = (moderngl.LINEAR, moderngl.LINEAR)

        sw, sh = screen
        x0 = px / sw * 2 - 1
        x1 = (px + w) / sw * 2 - 1
        y0 = 1 - py / sh * 2
        y1 = 1 - (py + h) / sh * 2
        quad = np.array([
            x0, y0, 0.0, 1.0,
            x0, y1, 0.0, 0.0,
            x1, y1, 1.0, 0.0,
            x0, y0, 0.0, 1.0,
            x1, y1, 1.0, 0.0,
            x1, y0, 1.0, 1.0,
        ], dtype="f4")
        vbo = self.ctx.buffer(quad.tobytes())
        vao = self.ctx.vertex_array(self.prog, [(vbo, "2f 2f", "in_pos", "in_uv")])

        self.ctx.disable(moderngl.DEPTH_TEST)
        self.ctx.enable(moderngl.BLEND)
        tex.use(0)
        self.prog["tex"].value = 0
        vao.render()
        self.ctx.disable(moderngl.BLEND)
        self.ctx.enable(moderngl.DEPTH_TEST)

        vao.release()
        vbo.release()
        tex.release()

    def draw_centered(self, text, screen, color, font):
        surf = font.render(text, True, color)
        w, h = surf.get_size()
        sw, sh = screen
        self.draw_text(text, (sw - w) // 2, (sh - h) // 2, screen, color, font)


def new_game(renderer, tex_arrays):
    """Generate a maze, verify it, build geometry, and return game state."""
    grid = maze.generate_maze(MAZE_N)
    if maze.verify_bfs(grid, MAZE_N):
        print(f"BFS: all {MAZE_N * MAZE_N} cells reachable")
    else:
        print("BFS: FAILED — regenerating")
        return new_game(renderer, tex_arrays)

    start, end = maze.pick_start_and_end(grid, MAZE_N)
    walls, floor, ceil = geometry.build_mesh(grid)
    pillar = geometry.build_pillar(end)
    renderer.set_world(
        {"walls": walls, "floor": floor, "ceil": ceil, "pillar": pillar},
        tex_arrays,
    )
    player = Player(start)
    print(f"start(z,x)={start}  exit(z,x)={end}")
    return grid, player, end


def main():
    pygame.init()
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MAJOR_VERSION, 3)
    pygame.display.gl_set_attribute(pygame.GL_CONTEXT_MINOR_VERSION, 3)
    pygame.display.gl_set_attribute(
        pygame.GL_CONTEXT_PROFILE_MASK, pygame.GL_CONTEXT_PROFILE_CORE
    )
    pygame.display.gl_set_attribute(pygame.GL_DEPTH_SIZE, 24)
    try:
        pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF, vsync=1)
    except TypeError:
        pygame.display.set_mode(WINDOW_SIZE, pygame.OPENGL | pygame.DOUBLEBUF)
    pygame.display.set_caption("3D Maze")

    renderer = Renderer()
    overlay = Overlay(renderer.ctx)

    # Texture arrays are static; generate once and reuse for every maze.
    tex_arrays = {
        "walls": textures.brick(),
        "floor": textures.floor_stone(),
        "ceil": textures.ceiling(),
        "pillar": textures.goal(),
    }
    flag_tex = textures.flag()
    flags = FlagSet()

    def reset_flags():
        flags.clear()
        renderer.set_flags(geometry.build_flags(flags.cells), flag_tex)

    grid, player, end = new_game(renderer, tex_arrays)
    reset_flags()
    clock = pygame.time.Clock()
    won = False

    while True:
        dt = clock.tick(60) / 1000.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    grid, player, end = new_game(renderer, tex_arrays)
                    reset_flags()
                    won = False
                if event.key == pygame.K_SPACE and not won:
                    if flags.toggle(player.cell) != "full":
                        renderer.set_flags(geometry.build_flags(flags.cells), flag_tex)

        keys = pygame.key.get_pressed()
        if not won:
            intents = {
                "forward": keys[pygame.K_w],
                "back": keys[pygame.K_s],
                "left": keys[pygame.K_a],
                "right": keys[pygame.K_d],
            }
            player.update(dt, intents, grid)
            if player.cell == end:
                won = True

        renderer.render(WINDOW_SIZE, (player.x, player.y, player.z), player.yaw)

        overlay.draw_text("WASD move/turn   Space flag   R new maze   Esc quit",
                          12, 10, WINDOW_SIZE)
        overlay.draw_text("Find the glowing exit pillar", 12, 38, WINDOW_SIZE,
                          color=(180, 255, 200))

        # Bottom-right: remaining flags the player can still place.
        flag_text = f"Flags: {flags.remaining}/{FlagSet.MAX}"
        fw, fh = overlay.font.size(flag_text)
        overlay.draw_text(flag_text, WINDOW_SIZE[0] - fw - 12, WINDOW_SIZE[1] - fh - 12,
                          WINDOW_SIZE, color=(255, 180, 90))

        if won:
            overlay.draw_centered("You reached the exit!", WINDOW_SIZE,
                                  (120, 255, 150), overlay.big)
            overlay.draw_text("Press R for a new maze", WINDOW_SIZE[0] // 2 - 130,
                              WINDOW_SIZE[1] // 2 + 50, WINDOW_SIZE, (235, 235, 245))

        pygame.display.flip()


if __name__ == "__main__":
    main()
