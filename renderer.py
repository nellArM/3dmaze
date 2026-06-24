"""moderngl-based renderer: shaders, textures, matrices, draw loop.

Attaches to the OpenGL context created by pygame.  Renders the batched maze
geometry (walls / floor / ceiling) plus an emissive exit pillar, with simple
directional lighting and exponential distance fog for the screensaver mood.
"""

import math
import numpy as np
import moderngl

# --- Tunables ---------------------------------------------------------------
FOG_COLOR = (0.04, 0.05, 0.08)
FOG_DENSITY = 0.085
LIGHT_DIR = (-0.4, -1.0, -0.3)


# --- Matrix helpers (row-major, used as M @ v) ------------------------------
def perspective(fovy_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fovy_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float32)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def look_at(eye, center, up):
    eye = np.asarray(eye, dtype=np.float64)
    center = np.asarray(center, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = center - eye
    f /= np.linalg.norm(f)
    s = np.cross(f, up)
    s /= np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.eye(4, dtype=np.float32)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


VERT_SHADER = """
#version 330
in vec3 in_pos;
in vec3 in_norm;
in vec2 in_uv;
uniform mat4 mvp;
out vec3 v_norm;
out vec2 v_uv;
out vec3 v_world;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_world = in_pos;
    v_norm = in_norm;
    v_uv = in_uv;
}
"""

FRAG_SHADER = """
#version 330
in vec3 v_norm;
in vec2 v_uv;
in vec3 v_world;
uniform sampler2D tex;
uniform vec3 cam_pos;
uniform vec3 light_dir;
uniform vec3 fog_color;
uniform float fog_density;
uniform int emissive;
out vec4 frag;
void main() {
    vec3 base = texture(tex, v_uv).rgb;
    vec3 color;
    if (emissive == 1) {
        color = base;
    } else {
        vec3 n = normalize(v_norm);
        float diff = max(dot(n, normalize(-light_dir)), 0.0);
        color = base * (0.45 + 0.75 * diff);
    }
    float dist = length(v_world - cam_pos);
    float fog = clamp(1.0 - exp(-fog_density * dist), 0.0, 1.0);
    frag = vec4(mix(color, fog_color, fog), 1.0);
}
"""


class Renderer:
    def __init__(self):
        self.ctx = moderngl.create_context()
        self.ctx.enable(moderngl.DEPTH_TEST)
        self.prog = self.ctx.program(vertex_shader=VERT_SHADER, fragment_shader=FRAG_SHADER)
        self.prog["light_dir"].value = LIGHT_DIR
        self.prog["fog_color"].value = FOG_COLOR
        self.prog["fog_density"].value = FOG_DENSITY
        self._objects = []   # static scene: list of (vao, texture, emissive)
        self._flag_obj = None  # dynamic flag batch: (vao, texture) or None

    def make_texture(self, img):
        """Upload an (H, W, 3) uint8 numpy array as a repeating, mipmapped texture."""
        h, w = img.shape[:2]
        tex = self.ctx.texture((w, h), 3, np.ascontiguousarray(img).tobytes())
        tex.build_mipmaps()
        tex.filter = (moderngl.LINEAR_MIPMAP_LINEAR, moderngl.LINEAR)
        tex.repeat_x = True
        tex.repeat_y = True
        try:
            tex.anisotropy = 8.0
        except Exception:
            pass
        return tex

    def _make_vao(self, verts):
        vbo = self.ctx.buffer(np.ascontiguousarray(verts, dtype="f4").tobytes())
        return self.ctx.vertex_array(
            self.prog, [(vbo, "3f 3f 2f", "in_pos", "in_norm", "in_uv")]
        )

    def set_world(self, meshes, textures):
        """Replace the scene.

        ``meshes`` = dict with keys ``walls``, ``floor``, ``ceil``, ``pillar``
        mapping to (N, 8) vertex arrays.  ``textures`` = dict with the same keys
        mapping to uint8 image arrays.
        """
        # Release previous GPU resources.
        for vao, tex, _ in self._objects:
            vao.release()
            tex.release()
        self._objects = []
        for key in ("walls", "floor", "ceil", "pillar"):
            verts = meshes.get(key)
            if verts is None or len(verts) == 0:
                continue
            vao = self._make_vao(verts)
            tex = self.make_texture(textures[key])
            emissive = 1 if key == "pillar" else 0
            self._objects.append((vao, tex, emissive))

    def set_flags(self, verts, tex_image):
        """Replace the dynamic flag batch. ``verts`` is an (N, 8) array (may be empty)."""
        if self._flag_obj is not None:
            vao, tex = self._flag_obj
            vao.release()
            tex.release()
            self._flag_obj = None
        if verts is not None and len(verts) > 0:
            self._flag_obj = (self._make_vao(verts), self.make_texture(tex_image))

    def render(self, size, eye, yaw):
        """Clear and draw the scene from camera at ``eye`` (x, y, z) facing ``yaw``."""
        w, h = size
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.clear(*FOG_COLOR, 1.0)

        ex, ey, ez = eye
        fwd = (ex + math.sin(yaw), ey, ez + math.cos(yaw))
        proj = perspective(70.0, w / max(h, 1), 0.05, 60.0)
        view = look_at(eye, fwd, (0.0, 1.0, 0.0))
        mvp = proj @ view
        # GLSL mat4 is column-major; upload the transpose so `mvp * v` is correct.
        self.prog["mvp"].write(np.ascontiguousarray(mvp.T).astype("f4").tobytes())
        self.prog["cam_pos"].value = (ex, ey, ez)

        for vao, tex, emissive in self._objects:
            tex.use(0)
            self.prog["tex"].value = 0
            self.prog["emissive"].value = emissive
            vao.render()

        # Dynamic flag markers, drawn emissive so they show through the fog.
        if self._flag_obj is not None:
            vao, tex = self._flag_obj
            tex.use(0)
            self.prog["tex"].value = 0
            self.prog["emissive"].value = 1
            vao.render()
