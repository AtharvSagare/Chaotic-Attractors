import numpy as np
import pygame
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate

# ------------------ Halvorsen parameter ------------------
a = 1.4

dt = 0.0003
steps = 500000        # LONGER trajectory
skip = 1

# ------------------ RK4 Integrator ------------------
def halvorsen_rk4(x, y, z, dt):
    def f(x, y, z):
        return np.array([
            -a*x - 4*y - 4*z - y*y,
            -a*y - 4*z - 4*x - z*z,
            -a*z - 4*x - 4*y - x*x
        ])

    k1 = f(x, y, z)
    k2 = f(x + 0.5*dt*k1[0], y + 0.5*dt*k1[1], z + 0.5*dt*k1[2])
    k3 = f(x + 0.5*dt*k2[0], y + 0.5*dt*k2[1], z + 0.5*dt*k2[2])
    k4 = f(x + dt*k3[0], y + dt*k3[1], z + dt*k3[2])

    return (
        x + dt*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0]) / 6,
        y + dt*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1]) / 6,
        z + dt*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2]) / 6,
    )

# ------------------ Generate trajectory ------------------
x, y, z = 1.0, 0.0, 0.0
pts = []

for i in range(steps):
    x, y, z = halvorsen_rk4(x, y, z, dt)
    if i > skip:
        pts.append([x, y, z])

pts = np.array(pts, dtype=np.float32)

# Center + robust scaling (better range)
pts -= pts.mean(axis=0)
scale = np.percentile(np.linalg.norm(pts, axis=1), 98)
pts /= scale
pts *= 7.5

progress = np.linspace(0, 1, len(pts)).astype(np.float32)

# ------------------ Shaders ------------------
VERT = """
uniform mat4 u_model;
uniform mat4 u_view;
uniform mat4 u_proj;

attribute vec3 a_position;
attribute float a_progress;
varying float v_prog;

void main() {
    v_prog = a_progress;
    gl_Position = u_proj * u_view * u_model * vec4(a_position, 1.0);
}
"""

FRAG = """
precision highp float;
uniform float u_time;
varying float v_prog;

float hash(float n){ return fract(sin(n)*43758.5453123); }
float noise(float x){
    float i = floor(x);
    float f = fract(x);
    return mix(hash(i), hash(i+1.0), f);
}

void main() {
    float flicker = 0.85 + 0.15 * noise(v_prog * 80.0 + u_time);

    vec3 deep  = vec3(0.17, 0.05, 0.05);
    vec3 hot   = vec3(1.0, 0.25, 0.0);
    vec3 white = vec3(1.0, 0.95, 0.78);

    vec3 col = mix(deep, hot, smoothstep(0.0, 0.6, v_prog));
    col = mix(col, white, smoothstep(0.6, 1.0, v_prog));

    col *= flicker * 1.3;
    gl_FragColor = vec4(col, 1.0);
}
"""

class HalvorsenCanvas(app.Canvas):
    def __init__(self):
        super().__init__(size=(1200, 800),
                         title="Halvorsen Attractor",
                         keys='interactive')

        # ------------------ MUSIC ------------------
        pygame.mixer.init()
        pygame.mixer.music.load("quantum.mp3")
        pygame.mixer.music.play(-1)

        self.program = gloo.Program(VERT, FRAG)
        self.program['a_position'] = pts[:2]
        self.program['a_progress'] = progress[:2]

        self.zoom = -9.5
        self.program['u_view'] = translate((0, 0, self.zoom))
        self.program['u_proj'] = perspective(
            45.0, self.size[0] / self.size[1], 0.1, 100.0
        )

        # -------- CONSTANT SPEED CONTROL --------
        self.count = 4
        self.POINTS_PER_FRAME = 180   # 👈 SAME speed forever

        # Rotation control
        self.rot_x = 0.0
        self.rot_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.dragging = False
        self.last_pos = None

        gloo.set_state(clear_color='black',
                       blend=True,
                       blend_func=('src_alpha', 'one'),
                       line_width=1.6)

        self.timer = app.Timer('auto', self.on_timer, start=True)
        self.show()

    def on_draw(self, event):
        gloo.clear()
        self.program['a_position'] = pts[:self.count]
        self.program['a_progress'] = progress[:self.count]
        self.program.draw('line_strip')

    def on_mouse_wheel(self, event):
        self.zoom += event.delta[1] * 0.6
        self.zoom = np.clip(self.zoom, -25.0, -3.0)
        self.program['u_view'] = translate((0, 0, self.zoom))
        self.update()

    def on_mouse_press(self, event):
        if event.button == 1:
            self.dragging = True
            self.last_pos = event.pos

    def on_mouse_release(self, event):
        self.dragging = False

    def on_mouse_move(self, event):
        if not self.dragging:
            return
        dx = event.pos[0] - self.last_pos[0]
        dy = event.pos[1] - self.last_pos[1]
        self.vel_x += dy * 0.12
        self.vel_y += dx * 0.12
        self.last_pos = event.pos

    def on_timer(self, event):
        self.program['u_time'] = event.elapsed

        # CONSTANT visual speed
        self.count = min(len(pts), self.count + self.POINTS_PER_FRAME)

        self.vel_x *= 0.92
        self.vel_y *= 0.92
        self.rot_x += self.vel_x
        self.rot_y += self.vel_y + 0.018

        model = np.eye(4, dtype=np.float32)
        model = rotate(self.rot_x, (1, 0, 0)) @ model
        model = rotate(self.rot_y, (0, 1, 0)) @ model
        self.program['u_model'] = model

        self.update()

    def on_key_press(self, event):
        if event.key == 'Space':
            if pygame.mixer.music.get_busy():
                pygame.mixer.music.pause()
            else:
                pygame.mixer.music.unpause()

# ------------------ Run ------------------
if __name__ == "__main__":
    c = HalvorsenCanvas()
    app.run()
