import numpy as np
from vispy import app, gloo
from vispy.util.transforms import perspective, translate, rotate

# ------------------ Lorenz parameters ------------------
sigma = 10.0
beta  = 8.0 / 3.0
rho   = 28.0

dt = 0.005
steps = 50000
skip = 100

# ------------------ RK4 Integrator ------------------
def lorenz_rk4(x, y, z, dt):
    def f(x, y, z):
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z
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
x, y, z = -10.0, -10.0, 25.0
pts = []

for i in range(steps):
    x, y, z = lorenz_rk4(x, y, z, dt)
    if i > skip:
        pts.append([x, y, z])

pts = np.array(pts, dtype=np.float32)
pts -= pts.mean(axis=0)
pts /= np.max(np.linalg.norm(pts, axis=1))
pts *= 3.5

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

class LorenzCanvas(app.Canvas):
    def __init__(self):
        super().__init__(size=(1200, 800),
                         title="Lorenz Attractor — Smooth Formation",
                         keys='interactive')

        self.program = gloo.Program(VERT, FRAG)
        self.program['a_position'] = pts[:2]   # start with first 2 points
        self.program['a_progress'] = progress[:2]

        self.program['u_view'] = translate((0, 0, -8))
        self.program['u_proj'] = perspective(
            45.0, self.size[0] / self.size[1], 0.1, 100.0
        )

        self.t = 0.0
        self.build = 0.0
        self.count = 2  # initial points

        self.rot_x = 0.0
        self.rot_y = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0
        self.dragging = False
        self.last_pos = None

        gloo.set_state(clear_color='black')
        self.timer = app.Timer('auto', self.on_timer, start=True)
        self.show()

    def on_draw(self, event):
        gloo.clear()
        # slice arrays to show only the built portion
        self.program['a_position'] = pts[:self.count]
        self.program['a_progress'] = progress[:self.count]
        self.program.draw('line_strip')

    def on_mouse_press(self, event):
        if event.button == 1:
            self.dragging = True
            self.last_pos = event.pos

    def on_mouse_release(self, event):
        self.dragging = False
        self.last_pos = None

    def on_mouse_move(self, event):
        if not self.dragging:
            return

        dx = event.pos[0] - self.last_pos[0]
        dy = event.pos[1] - self.last_pos[1]

        self.vel_x += dy * 0.12
        self.vel_y += dx * 0.12
        self.last_pos = event.pos

    def on_timer(self, event):
        self.t += 0.016
        self.program['u_time'] = self.t

        # grow attractor smoothly
        self.build = min(1.0, self.build + 0.0015)
        self.count = max(2, int(self.build * len(pts)))

        self.vel_x *= 0.92
        self.vel_y *= 0.92

        self.rot_x += self.vel_x
        self.rot_y += self.vel_y + 0.02

        model = np.eye(4, dtype=np.float32)
        model = rotate(self.rot_x, (1, 0, 0)) @ model
        model = rotate(self.rot_y, (0, 1, 0)) @ model

        self.program['u_model'] = model
        self.update()

    def on_resize(self, event):
        w, h = event.size
        gloo.set_viewport(0, 0, w, h)
        self.program['u_proj'] = perspective(45.0, w / h, 0.1, 100.0)

# ------------------ Run ------------------
if __name__ == "__main__":
    c = LorenzCanvas()
    app.run()
