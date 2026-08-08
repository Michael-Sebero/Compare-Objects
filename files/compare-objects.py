#!/usr/bin/env python3

import math
import re

try:
    import tkinter as tk
    from tkinter import ttk
except ImportError:
    import sys
    print("Compare Objects needs tkinter, which isn't installed for this Python.")
    print()
    print("On Linux, tkinter is usually a separate system package:")
    print("    sudo apt install python3-tk        (Debian / Ubuntu)")
    print("    sudo dnf install python3-tkinter   (Fedora)")
    print("On macOS/Windows, installing Python from python.org includes it --")
    print("if you installed Python a different way, try that instead.")
    sys.exit(1)


# ====================================================================
# 3D math — camera, projection, ray-plane intersection
# ====================================================================


def vec3(x, y, z):
    return (float(x), float(y), float(z))


def add(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def scale(a, s):
    return (a[0] * s, a[1] * s, a[2] * s)


def dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def cross(a, b):
    return (
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    )


def length(a):
    return math.sqrt(dot(a, a))


def normalize(a):
    l = length(a)
    if l < 1e-9:
        return (0.0, 0.0, 0.0)
    return scale(a, 1.0 / l)


class OrbitCamera:
    """
    Orbit camera: looks at `target` from a point on a sphere of radius
    `distance`, at azimuth `theta` and elevation `phi` (both radians).
    World "up" is +Z (matches the app's z=height convention).
    """

    def __init__(self, target, theta, phi, distance, fov_deg):
        self.target = target
        self.theta = theta
        self.phi = phi
        self.distance = distance
        self.fov = math.radians(fov_deg)

    @property
    def eye(self):
        cp, sp = math.cos(self.phi), math.sin(self.phi)
        ct, st = math.cos(self.theta), math.sin(self.theta)
        return (
            self.target[0] + self.distance * cp * ct,
            self.target[1] + self.distance * cp * st,
            self.target[2] + self.distance * sp,
        )

    @property
    def basis(self):
        """Right-handed camera basis: forward points from eye to target."""
        eye = self.eye
        forward = normalize(sub(self.target, eye))
        world_up = (0.0, 0.0, 1.0)
        right = cross(forward, world_up)
        if length(right) < 1e-6:
            # Looking almost straight up/down; world-up is ambiguous, pick
            # a stable fallback so basis vectors don't degenerate.
            right = (1.0, 0.0, 0.0)
        right = normalize(right)
        up = normalize(cross(right, forward))
        return eye, forward, right, up


def project(camera, viewport_w, viewport_h, p):
    """World point -> (screen_x, screen_y, depth), or None if behind camera."""
    eye, forward, right, up = camera.basis
    rel = sub(p, eye)
    cam_x = dot(rel, right)
    cam_y = dot(rel, up)
    cam_z = dot(rel, forward)
    if cam_z <= 1e-6:
        return None
    f = (viewport_h / 2) / math.tan(camera.fov / 2)
    return (
        viewport_w / 2 + (cam_x / cam_z) * f,
        viewport_h / 2 - (cam_y / cam_z) * f,
        cam_z,
    )


def screen_to_ray(camera, viewport_w, viewport_h, sx, sy):
    """Screen point -> (ray_origin, ray_direction), direction normalized."""
    eye, forward, right, up = camera.basis
    f = (viewport_h / 2) / math.tan(camera.fov / 2)
    cam_x = sx - viewport_w / 2
    cam_y = -(sy - viewport_h / 2)
    direction = normalize(
        add(add(scale(right, cam_x), scale(up, cam_y)), scale(forward, f))
    )
    return eye, direction


def ray_plane_intersect(ray_origin, ray_dir, plane_point, plane_normal):
    """Intersect a ray with a plane (point + normal). None if parallel/behind."""
    denom = dot(ray_dir, plane_normal)
    if abs(denom) < 1e-9:
        return None
    t = dot(sub(plane_point, ray_origin), plane_normal) / denom
    if t < 0:
        return None
    return add(ray_origin, scale(ray_dir, t))


# ====================================================================
# Geometry — cuboid faces, convex hull, point-in-polygon hit testing
# ====================================================================
def cuboid_faces(b):
    """
    Six faces of an axis-aligned box (dict with xmin/xmax/ymin/ymax/
    zmin/zmax), each as (name, normal, [4 vertices]). Vertex order gives
    the correct OUTWARD normal via (v1-v0) x (v2-v0) -- verified
    computationally against expected normals before this was written.
    """
    xmin, xmax = b["xmin"], b["xmax"]
    ymin, ymax = b["ymin"], b["ymax"]
    zmin, zmax = b["zmin"], b["zmax"]
    p = {
        "000": (xmin, ymin, zmin), "100": (xmax, ymin, zmin),
        "110": (xmax, ymax, zmin), "010": (xmin, ymax, zmin),
        "001": (xmin, ymin, zmax), "101": (xmax, ymin, zmax),
        "111": (xmax, ymax, zmax), "011": (xmin, ymax, zmax),
    }
    return [
        {"name": "top", "normal": (0, 0, 1), "pts": [p["001"], p["101"], p["111"], p["011"]]},
        {"name": "bottom", "normal": (0, 0, -1), "pts": [p["000"], p["010"], p["110"], p["100"]]},
        {"name": "front", "normal": (0, -1, 0), "pts": [p["000"], p["100"], p["101"], p["001"]]},
        {"name": "back", "normal": (0, 1, 0), "pts": [p["010"], p["011"], p["111"], p["110"]]},
        {"name": "left", "normal": (-1, 0, 0), "pts": [p["000"], p["001"], p["011"], p["010"]]},
        {"name": "right", "normal": (1, 0, 0), "pts": [p["100"], p["110"], p["111"], p["101"]]},
    ]


def all_corners(b):
    xmin, xmax = b["xmin"], b["xmax"]
    ymin, ymax = b["ymin"], b["ymax"]
    zmin, zmax = b["zmin"], b["zmax"]
    return [
        (x, y, z)
        for x in (xmin, xmax)
        for y in (ymin, ymax)
        for z in (zmin, zmax)
    ]


def convex_hull(points):
    """
    Andrew's monotone-chain convex hull. points: [(x,y), ...]. Returns
    hull in CCW order.
    """
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def point_in_convex_polygon(pt, hull):
    """True if `pt` is inside (or on the boundary of) the convex polygon."""
    if len(hull) < 3:
        return False
    sign = 0
    n = len(hull)
    for i in range(n):
        a = hull[i]
        b = hull[(i + 1) % n]
        cr = (b[0] - a[0]) * (pt[1] - a[1]) - (b[1] - a[1]) * (pt[0] - a[0])
        if cr != 0:
            s = 1 if cr > 0 else -1
            if sign == 0:
                sign = s
            elif s != sign:
                return False
    return True


def polygon_area(hull):
    a = 0.0
    n = len(hull)
    for i in range(n):
        p = hull[i]
        q = hull[(i + 1) % n]
        a += p[0] * q[1] - q[0] * p[1]
    return abs(a) / 2


# ====================================================================
# Scene model — blocks, units, layout
# ====================================================================

UNIT_TO_M = {
    "mm": 0.001, "millimeter": 0.001, "millimeters": 0.001,
    "cm": 0.01, "centimeter": 0.01, "centimeters": 0.01,
    "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0, "metres": 1.0,
    "in": 0.0254, "inch": 0.0254, "inches": 0.0254, '"': 0.0254,
    "ft": 0.3048, "foot": 0.3048, "feet": 0.3048, "'": 0.3048,
    "yd": 0.9144, "yard": 0.9144, "yards": 0.9144,
}

_FT_IN_RE = re.compile(
    r"^(?P<ft>-?\d+(?:\.\d+)?)\s*(?:'|ft|feet|foot)\s*"
    r"(?:(?P<in>\d+(?:\.\d+)?)\s*(?:\"|in|inch|inches)?)?$"
)
_NUM_UNIT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*([a-zA-Z\"']*)$")

PALETTE = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
]


def to_meters(value, unit="m"):
    """
    Convert a length to meters.
    - If `value` is a number, it's interpreted in `unit`.
    - If `value` is a string, it may carry its own unit
      ("6ft", "1.8m", "5'10\""); otherwise `unit` is the default.
    """
    if isinstance(value, (int, float)):
        u = unit.strip().lower()
        if u not in UNIT_TO_M:
            raise ValueError(f"Unknown unit: {unit!r}")
        return float(value) * UNIT_TO_M[u]

    s = str(value).strip()
    if not s:
        raise ValueError("Empty length value")
    sl = s.lower()

    m = _FT_IN_RE.match(sl)
    if m:
        ft = float(m.group("ft"))
        inch = float(m.group("in")) if m.group("in") else 0.0
        return ft * UNIT_TO_M["ft"] + inch * UNIT_TO_M["in"]

    m = _NUM_UNIT_RE.match(sl)
    if m:
        num = float(m.group(1))
        u = m.group(2).strip().lower() or unit.strip().lower()
        if u not in UNIT_TO_M:
            raise ValueError(f"Unknown unit {u!r} in {value!r}")
        return num * UNIT_TO_M[u]

    raise ValueError(f"Could not parse length: {value!r}")


def meters_to_ft_in(m):
    total_in = m / UNIT_TO_M["in"]
    ft = int(total_in // 12)
    inch = total_in - ft * 12
    return ft, inch


def format_length(m, unit_mode="m"):
    if unit_mode == "m":
        return f"{m:.2f} m" if m >= 1 else f"{m * 100:.1f} cm"
    ft, inch = meters_to_ft_in(m)
    return f'{inch:.1f}"' if ft == 0 else f'{ft}\'{inch:.1f}"'


class Block:
    """
    A rectangular object with a real-world size (w=x, d=y, h=z, all in
    meters) and a position (x, y, z = the block's bottom-center point,
    so z=0 means resting on the ground).
    """

    _next_color = 0
    _next_id = 1

    def __init__(self, name, w, d, h, color=None):
        self.id = Block._next_id
        Block._next_id += 1
        self.name = name
        self.w = w
        self.d = d
        self.h = h
        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        if color is None:
            color = PALETTE[Block._next_color % len(PALETTE)]
            Block._next_color += 1
        self.color = color

    @classmethod
    def reset_color_cursor(cls):
        cls._next_color = 0

    def bounds(self):
        return {
            "xmin": self.x - self.w / 2, "xmax": self.x + self.w / 2,
            "ymin": self.y - self.d / 2, "ymax": self.y + self.d / 2,
            "zmin": self.z, "zmax": self.z + self.h,
        }

    def dims_label(self, unit_mode="m"):
        return (f"{format_length(self.w, unit_mode)} × "
                f"{format_length(self.d, unit_mode)} × "
                f"{format_length(self.h, unit_mode)}")


def arrange_side_by_side(blocks):
    """Place blocks in a row along X, centered on Y=0, resting on Z=0."""
    if not blocks:
        return
    # Use each block's larger of width/depth so a "deep" object (e.g. a
    # bicycle's wheel-to-wheel length) still gets a generous gap even
    # though depth isn't the axis it's laid out along -- otherwise a deep
    # object can visually swallow a small neighbor from an angled camera.
    avg_footprint = sum(max(b.w, b.d) for b in blocks) / len(blocks)
    gap = max(avg_footprint * 0.2, 0.08)
    cursor = 0.0
    for b in blocks:
        b.x = cursor + b.w / 2
        b.y = 0.0
        b.z = 0.0
        cursor += b.w + gap
    total_width = cursor - gap
    offset = total_width / 2
    for b in blocks:
        b.x -= offset


def place_new_block_next_to_others(new_block, others):
    if not others:
        new_block.x = new_block.y = new_block.z = 0.0
        return
    max_x = max(b.bounds()["xmax"] for b in others)
    avg_footprint = sum(max(b.w, b.d) for b in others) / len(others)
    gap = max(avg_footprint * 0.2, 0.08)
    new_block.x = max_x + gap + new_block.w / 2
    new_block.y = 0.0
    new_block.z = 0.0


# ====================================================================
# Application — rendering, interaction, panel UI
# ====================================================================


# ---- design tokens (matching the earlier web build) ----
BG = "#161a22"
BG_PANEL = "#1c212b"
BG_ELEVATED = "#242b38"
BG_ELEVATED_HOVER = "#2c3444"
LINE = "#323b4d"
LINE_SOFT = "#2a3140"
TEXT = "#eef1f7"
TEXT_DIM = "#98a1b5"
TEXT_FAINT = "#626d84"
ACCENT = "#e6a23c"
ACCENT_STRONG = "#f5b74f"
ACCENT_INK = "#241a05"
DANGER = "#d1654f"
DANGER_STRONG = "#e57c65"
FONT_UI = ("Segoe UI", 10)
FONT_UI_SM = ("Segoe UI", 9)
FONT_UI_BOLD = ("Segoe UI", 11, "bold")
FONT_LABEL_HEAD = ("Segoe UI", 9, "bold")
FONT_MONO = ("Consolas", 9)
FONT_MONO_SM = ("Consolas", 8)

MIN_PHI, MAX_PHI = 0.06, math.pi / 2 - 0.03
MIN_DIST, MAX_DIST = 0.25, 800.0
LIGHT_DIR = normalize(vec3(0.45, -0.55, 0.72))


def shade_color(hex_color, factor):
    hex_color = hex_color.lstrip("#")
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)

    def c(v):
        return max(0, min(255, round(v * factor)))

    return f"#{c(r):02x}{c(g):02x}{c(b):02x}"


def nice_step(rng):
    target_lines = 9
    raw = rng / target_lines if rng else 1
    mag = 10 ** math.floor(math.log10(raw)) if raw > 0 else 1
    norm = raw / mag
    if norm < 1.5:
        step = 1
    elif norm < 3.5:
        step = 2
    elif norm < 7.5:
        step = 5
    else:
        step = 10
    return step * mag


class ScrollableFrame(tk.Frame):
    """A vertically-scrollable container (Tk has no built-in equivalent)."""

    def __init__(self, container, width, **kwargs):
        super().__init__(container, bg=BG_PANEL, **kwargs)
        self._canvas = tk.Canvas(self, bg=BG_PANEL, highlightthickness=0, width=width)
        scrollbar = tk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self.body = tk.Frame(self._canvas, bg=BG_PANEL)

        self.body.bind("<Configure>", lambda e: self._canvas.configure(scrollregion=self._canvas.bbox("all")))
        self._window = self._canvas.create_window((0, 0), window=self.body, anchor="nw")
        self._canvas.bind("<Configure>", lambda e: self._canvas.itemconfig(self._window, width=e.width))
        self._canvas.configure(yscrollcommand=scrollbar.set)

        self._canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        self._canvas.bind("<Enter>", lambda e: self._canvas.bind_all("<MouseWheel>", self._on_wheel))
        self._canvas.bind("<Leave>", lambda e: self._canvas.unbind_all("<MouseWheel>"))

    def _on_wheel(self, event):
        self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")


class App:
    def __init__(self, root, blocks):
        self.root = root
        self.blocks = blocks
        self.selected_id = blocks[0].id if blocks else None
        self.unit_mode = "m"
        self.camera = OrbitCamera(vec3(0, 0, 0), -1.0, 0.45, 10, 42)
        self.drag_state = None

        self._build_ui()
        self.fit_camera_to_scene()
        self.canvas.bind("<Configure>", lambda e: self.render())
        self.refresh_panel()
        self.root.after(50, self.render)

    # ================================================================
    # layout
    # ================================================================
    def _build_ui(self):
        self.root.configure(bg=BG)
        outer = tk.Frame(self.root, bg=BG)
        outer.pack(fill="both", expand=True)

        # ---- 3D viewport ----
        canvas_wrap = tk.Frame(outer, bg=BG)
        canvas_wrap.pack(side="left", fill="both", expand=True)
        self.canvas = tk.Canvas(canvas_wrap, bg=BG, highlightthickness=0, cursor="arrow")
        self.canvas.pack(fill="both", expand=True)

        hint = ("drag (or middle-drag) empty space to orbit  ·  drag a block to move it  ·  shift+drag to raise/lower\n"
                "arrow keys / WASD to nudge on the grid  ·  Q / E to raise/lower  ·  scroll to zoom")
        tk.Label(canvas_wrap, text=hint, bg="#161a22", fg=TEXT_FAINT, font=FONT_MONO_SM, anchor="w",
                 justify="left").place(x=14, rely=1.0, y=-36)

        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<ButtonPress-2>", self.on_middle_press)   # middle-click: orbit, anywhere
        self.canvas.bind("<B2-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-2>", self.on_release)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<MouseWheel>", self.on_wheel)       # Windows / macOS
        self.canvas.bind("<Button-4>", lambda e: self.zoom_step(1))   # Linux scroll up
        self.canvas.bind("<Button-5>", lambda e: self.zoom_step(-1))  # Linux scroll down

        self.root.bind("<Key>", self.on_key)

        # ---- side panel ----
        panel_outer = tk.Frame(outer, bg=BG_PANEL, width=340)
        panel_outer.pack(side="right", fill="y")
        panel_outer.pack_propagate(False)

        scroller = ScrollableFrame(panel_outer, width=340)
        scroller.pack(fill="both", expand=True)
        p = scroller.body

        tk.Label(p, text="Compare Objects", bg=BG_PANEL, fg=TEXT, font=FONT_UI_BOLD).pack(
            anchor="w", padx=16, pady=(16, 12)
        )

        tk.Label(p, text="OBJECTS", bg=BG_PANEL, fg=TEXT_FAINT, font=FONT_LABEL_HEAD).pack(
            anchor="w", padx=16
        )
        self.list_frame = tk.Frame(p, bg=BG_PANEL)
        self.list_frame.pack(fill="x", padx=12, pady=(6, 14))

        self.detail_frame = tk.Frame(p, bg=BG_PANEL)
        self.detail_frame.pack(fill="x", padx=16, pady=(0, 14))

        tk.Label(p, text="ADD OBJECT", bg=BG_PANEL, fg=TEXT_FAINT, font=FONT_LABEL_HEAD).pack(
            anchor="w", padx=16
        )
        self._build_add_form(p)

        btn_row = tk.Frame(p, bg=BG_PANEL)
        btn_row.pack(fill="x", padx=16, pady=(14, 20))
        self._make_button(btn_row, "Reset layout", self.reset_layout).pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.units_btn = self._make_button(btn_row, "Units: m", self.toggle_units)
        self.units_btn.pack(side="left", fill="x", expand=True, padx=(4, 0))

    def _make_button(self, parent, text, command, accent=False):
        bg = ACCENT if accent else BG_ELEVATED
        fg = ACCENT_INK if accent else TEXT
        b = tk.Button(
            parent, text=text, command=command, bg=bg, fg=fg,
            activebackground=ACCENT_STRONG if accent else BG_ELEVATED_HOVER,
            activeforeground=fg, font=FONT_UI_SM, relief="flat", bd=0,
            padx=10, pady=7, cursor="hand2",
        )
        return b

    def _build_add_form(self, parent):
        form = tk.Frame(parent, bg=BG_PANEL)
        form.pack(fill="x", padx=16, pady=(6, 0))

        self.add_name_var = tk.StringVar()
        e = tk.Entry(form, textvariable=self.add_name_var, bg=BG, fg=TEXT, insertbackground=TEXT,
                     relief="flat", font=FONT_MONO, highlightthickness=1,
                     highlightbackground=LINE, highlightcolor=ACCENT)
        e.insert(0, "")
        e.pack(fill="x", pady=(0, 6))
        e.bind("<FocusIn>", lambda ev: (e.delete(0, "end") if e.get() == "" else None))
        self._add_placeholder(e, "Name")

        dims = tk.Frame(form, bg=BG_PANEL)
        dims.pack(fill="x", pady=(0, 6))
        self.add_x_var, self.add_y_var, self.add_z_var = tk.StringVar(), tk.StringVar(), tk.StringVar()
        for col, (label, var) in enumerate((("X width", self.add_x_var), ("Y depth", self.add_y_var), ("Z height", self.add_z_var))):
            cell = tk.Frame(dims, bg=BG_PANEL)
            cell.grid(row=0, column=col, sticky="ew", padx=(0 if col == 0 else 4, 0))
            dims.grid_columnconfigure(col, weight=1)
            tk.Label(cell, text=label, bg=BG_PANEL, fg=TEXT_FAINT, font=FONT_MONO_SM).pack(anchor="w")
            ent = tk.Entry(cell, textvariable=var, bg=BG, fg=TEXT, insertbackground=TEXT,
                           relief="flat", font=FONT_MONO, highlightthickness=1,
                           highlightbackground=LINE, highlightcolor=ACCENT, width=8)
            ent.pack(fill="x")

        self.add_unit_var = tk.StringVar(value="m")
        unit_combo = ttk.Combobox(
            form, textvariable=self.add_unit_var, state="readonly",
            values=["m", "cm", "mm", "ft", "in", "yd"], font=FONT_MONO,
        )
        unit_combo.pack(fill="x", pady=(0, 8))

        self._add_name_entry = e
        self._make_button(form, "Add object", self.add_block_from_form, accent=True).pack(fill="x")

    def _add_placeholder(self, entry, text):
        entry.config(fg=TEXT_FAINT)
        entry.insert(0, text)

        def on_focus_in(_e):
            if entry.get() == text:
                entry.delete(0, "end")
                entry.config(fg=TEXT)

        def on_focus_out(_e):
            if not entry.get():
                entry.config(fg=TEXT_FAINT)
                entry.insert(0, text)

        entry.bind("<FocusIn>", on_focus_in)
        entry.bind("<FocusOut>", on_focus_out)

    # ================================================================
    # camera / bounds
    # ================================================================
    def scene_bounds(self):
        if not self.blocks:
            return {"minX": -1, "maxX": 1, "minY": -1, "maxY": 1, "minZ": 0, "maxZ": 1}
        xs0, xs1, ys0, ys1, zs0, zs1 = [], [], [], [], [], []
        for b in self.blocks:
            bd = b.bounds()
            xs0.append(bd["xmin"]); xs1.append(bd["xmax"])
            ys0.append(bd["ymin"]); ys1.append(bd["ymax"])
            zs0.append(bd["zmin"]); zs1.append(bd["zmax"])
        return {"minX": min(xs0), "maxX": max(xs1), "minY": min(ys0), "maxY": max(ys1),
                "minZ": min(zs0), "maxZ": max(zs1)}

    def fit_camera_to_scene(self):
        b = self.scene_bounds()
        cx, cy, cz = (b["minX"] + b["maxX"]) / 2, (b["minY"] + b["maxY"]) / 2, (b["minZ"] + b["maxZ"]) / 2
        diag = max(length(vec3(b["maxX"] - b["minX"], b["maxY"] - b["minY"], b["maxZ"] - b["minZ"])), 0.5)
        self.camera.target = vec3(cx, cy, cz)
        self.camera.distance = max(diag * 1.35 / math.tan(self.camera.fov / 2) * 0.62, diag * 1.1)

    def ensure_visible(self, block):
        center_dist = length(sub(vec3(block.x, block.y, block.z + block.h / 2), self.camera.target))
        if center_dist > self.camera.distance * 0.85:
            self.camera.distance = center_dist / 0.7

    def nudge_step(self):
        if not self.blocks:
            return 0.05
        m = max(max(b.w, b.d, b.h) for b in self.blocks)
        return max(m * 0.02, 0.01)

    def find_block(self, block_id):
        return next((b for b in self.blocks if b.id == block_id), None)

    # ================================================================
    # rendering
    # ================================================================
    def canvas_size(self):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        return (w if w > 1 else 800, h if h > 1 else 600)

    def draw_outlined_text(self, x, y, text, fill, font):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            self.canvas.create_text(x + dx, y + dy, text=text, fill="#0c0e14", font=font)
        self.canvas.create_text(x, y, text=text, fill=fill, font=font)

    def draw_ground_grid(self, w, h):
        b = self.scene_bounds()
        span_x, span_y = max(b["maxX"] - b["minX"], 1), max(b["maxY"] - b["minY"], 1)
        pad = max(span_x, span_y) * 0.9 + 1
        cx, cy = (b["minX"] + b["maxX"]) / 2, (b["minY"] + b["maxY"]) / 2
        x0, x1, y0, y1 = cx - pad, cx + pad, cy - pad, cy + pad
        step = nice_step(pad * 2)

        x = math.floor(x0 / step) * step
        while x <= x1:
            p1, p2 = project(self.camera, w, h, vec3(x, y0, 0)), project(self.camera, w, h, vec3(x, y1, 0))
            if p1 and p2:
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#2c2a1e", width=1)
            x += step
        y = math.floor(y0 / step) * step
        while y <= y1:
            p1, p2 = project(self.camera, w, h, vec3(x0, y, 0)), project(self.camera, w, h, vec3(x1, y, 0))
            if p1 and p2:
                self.canvas.create_line(p1[0], p1[1], p2[0], p2[1], fill="#2c2a1e", width=1)
            y += step

    def draw_block_faces(self, b, w, h):
        bnd = b.bounds()
        eye = self.camera.eye
        is_sel = self.selected_id == b.id
        for face in cuboid_faces(bnd):
            pts3 = face["pts"]
            cx = (pts3[0][0] + pts3[2][0]) / 2
            cy = (pts3[0][1] + pts3[2][1]) / 2
            cz = (pts3[0][2] + pts3[2][2]) / 2
            if dot(face["normal"], sub(eye, (cx, cy, cz))) <= 0:
                continue
            screen_pts = [project(self.camera, w, h, vec3(*p)) for p in pts3]
            if any(p is None for p in screen_pts):
                continue
            brightness = max(0.4, min(1.15, dot(face["normal"], LIGHT_DIR) * 0.75 + 0.55))
            flat = [c for p in screen_pts for c in (p[0], p[1])]
            self.canvas.create_polygon(
                *flat, fill=shade_color(b.color, brightness),
                outline=ACCENT_STRONG if is_sel else "#0a0c12", width=2.5 if is_sel else 1,
            )

    def draw_label(self, b, index, w, h):
        bnd = b.bounds()
        scr = project(self.camera, w, h, vec3(b.x, b.y, bnd["zmax"]))
        if not scr:
            return
        is_sel = self.selected_id == b.id
        self.draw_outlined_text(scr[0], scr[1] - 20, f"{index + 1}. {b.name}",
                                 ACCENT_STRONG if is_sel else TEXT, FONT_LABEL_HEAD)
        self.draw_outlined_text(scr[0], scr[1] - 7, b.dims_label(self.unit_mode), "#b7bfcf", FONT_MONO_SM)

    def render(self):
        w, h = self.canvas_size()
        self.canvas.delete("all")
        if not self.blocks:
            self.canvas.create_text(w / 2, h / 2, text="Nothing to compare yet — add an object to get started.",
                                     fill=TEXT_FAINT, font=FONT_UI)
            return
        self.draw_ground_grid(w, h)
        ordered = sorted(
            self.blocks,
            key=lambda b: -dot(sub(vec3(b.x, b.y, b.z + b.h / 2), self.camera.eye), self.camera.basis[1]),
        )
        for b in ordered:
            self.draw_block_faces(b, w, h)
        for i, b in enumerate(self.blocks):
            self.draw_label(b, i, w, h)

    # ================================================================
    # hit testing + pointer interaction
    # ================================================================
    def hit_test_blocks(self, mx, my):
        w, h = self.canvas_size()
        best, best_depth = None, float("inf")
        for b in self.blocks:
            corners = all_corners(b.bounds())
            screen_pts = [project(self.camera, w, h, vec3(*c)) for c in corners]
            screen_pts = [(p[0], p[1]) for p in screen_pts if p is not None]
            if len(screen_pts) < 3:
                continue
            hull = convex_hull(screen_pts)
            if point_in_convex_polygon((mx, my), hull):
                depth = dot(sub(vec3(b.x, b.y, b.z + b.h / 2), self.camera.eye), self.camera.basis[1])
                if depth < best_depth:
                    best_depth, best = depth, b
        return best

    def on_press(self, event):
        self.canvas.focus_set()
        hit = self.hit_test_blocks(event.x, event.y)
        shift = bool(event.state & 0x1)
        if hit:
            self.select_block(hit.id)
            if shift:
                self.drag_state = {"mode": "height", "block_id": hit.id, "last_y": event.y}
            else:
                w, h = self.canvas_size()
                origin, direction = screen_to_ray(self.camera, w, h, event.x, event.y)
                hp = ray_plane_intersect(origin, direction, vec3(0, 0, hit.z), vec3(0, 0, 1))
                offset = (hit.x - hp[0], hit.y - hp[1]) if hp else (0.0, 0.0)
                self.drag_state = {"mode": "move", "block_id": hit.id, "plane_z": hit.z, "offset": offset}
            self.canvas.config(cursor="fleur")
        else:
            self.drag_state = {"mode": "orbit", "last_x": event.x, "last_y": event.y}
            self.canvas.config(cursor="exchange")

    def on_middle_press(self, event):
        # Middle-click always orbits, regardless of what's under the
        # cursor -- a dedicated "camera button" like most 3D tools use,
        # so you never have to worry about accidentally grabbing a block
        # just to look around.
        self.canvas.focus_set()
        self.drag_state = {"mode": "orbit", "last_x": event.x, "last_y": event.y}
        self.canvas.config(cursor="exchange")

    def on_drag(self, event):
        if not self.drag_state:
            return
        mode = self.drag_state["mode"]
        w, h = self.canvas_size()
        if mode == "orbit":
            dx, dy = event.x - self.drag_state["last_x"], event.y - self.drag_state["last_y"]
            self.camera.theta -= dx * 0.007
            self.camera.phi = min(MAX_PHI, max(MIN_PHI, self.camera.phi + dy * 0.007))
            self.drag_state["last_x"], self.drag_state["last_y"] = event.x, event.y
            self.render()
        elif mode == "move":
            b = self.find_block(self.drag_state["block_id"])
            if not b:
                return
            origin, direction = screen_to_ray(self.camera, w, h, event.x, event.y)
            hp = ray_plane_intersect(origin, direction, vec3(0, 0, self.drag_state["plane_z"]), vec3(0, 0, 1))
            if hp:
                b.x = hp[0] + self.drag_state["offset"][0]
                b.y = hp[1] + self.drag_state["offset"][1]
                self.render()
                self.update_detail_fields_live()
        elif mode == "height":
            b = self.find_block(self.drag_state["block_id"])
            if not b:
                return
            # Incremental screen-delta control (not ray/plane intersection):
            # a plane derived from the camera's own basis becomes nearly
            # edge-on at some viewing angles, which makes ray-plane
            # intersection wildly oversensitive there. A fixed pixels-to-
            # meters scale, keyed off camera distance so it still feels
            # consistent at any zoom level, has no such blowup.
            dy = event.y - self.drag_state["last_y"]
            scale = self.camera.distance * 0.0015
            b.z = max(0.0, b.z - dy * scale)
            self.drag_state["last_y"] = event.y
            self.render()
            self.update_detail_fields_live()

    def on_release(self, _event):
        if self.drag_state:
            self.drag_state = None
            self.canvas.config(cursor="arrow")
            self.refresh_panel()

    def on_hover(self, event):
        if self.drag_state:
            return
        hit = self.hit_test_blocks(event.x, event.y)
        self.canvas.config(cursor="hand2" if hit else "arrow")

    def zoom_step(self, direction):
        factor = math.exp(-direction * 0.12)
        self.camera.distance = min(MAX_DIST, max(MIN_DIST, self.camera.distance * factor))
        self.render()

    def on_wheel(self, event):
        factor = math.exp(-event.delta * 0.001)
        self.camera.distance = min(MAX_DIST, max(MIN_DIST, self.camera.distance * factor))
        self.render()

    def on_key(self, event):
        focused = self.root.focus_get()
        if isinstance(focused, (tk.Entry, ttk.Entry, ttk.Combobox)):
            return
        key = event.keysym
        if key in ("Delete", "BackSpace"):
            if self.selected_id is not None:
                self.delete_block(self.selected_id)
        elif key in ("u", "U"):
            self.toggle_units()
        elif key in ("r", "R"):
            self.reset_layout()
        elif key in ("Left", "Right", "Up", "Down", "a", "A", "d", "D", "w", "W", "s", "S"):
            # Arrow keys and WASD are equivalent: both nudge the selected
            # block around on the grid (X/Y). W/Up = away, S/Down =
            # toward, A/Left = left, D/Right = right.
            b = self.find_block(self.selected_id)
            if not b:
                return
            step = self.nudge_step()
            if key in ("Left", "a", "A"):
                b.x -= step
            elif key in ("Right", "d", "D"):
                b.x += step
            elif key in ("Up", "w", "W"):
                b.y += step
            elif key in ("Down", "s", "S"):
                b.y -= step
            self.render()
            self.update_detail_fields_live()
        elif key in ("q", "Q"):
            b = self.find_block(self.selected_id)
            if b:
                b.z += self.nudge_step()
                self.render()
                self.update_detail_fields_live()
        elif key in ("e", "E"):
            b = self.find_block(self.selected_id)
            if b:
                b.z = max(0.0, b.z - self.nudge_step())
                self.render()
                self.update_detail_fields_live()

    # ================================================================
    # panel actions
    # ================================================================
    def select_block(self, block_id):
        self.selected_id = block_id
        self.refresh_panel()
        self.render()

    def delete_block(self, block_id):
        idx = next((i for i, b in enumerate(self.blocks) if b.id == block_id), None)
        if idx is None:
            return
        self.blocks.pop(idx)
        if self.selected_id == block_id:
            self.selected_id = self.blocks[min(idx, len(self.blocks) - 1)].id if self.blocks else None
        self.refresh_panel()
        self.render()

    def add_block_from_form(self):
        name = self.add_name_var.get().strip()
        if not name or name == "Name":
            name = f"Object {len(self.blocks) + 1}"
        unit = self.add_unit_var.get() or "m"
        try:
            xv = float(self.add_x_var.get())
            yv = float(self.add_y_var.get())
            zv = float(self.add_z_var.get())
            if not (xv > 0 and yv > 0 and zv > 0):
                raise ValueError
        except ValueError:
            self._flash_form_error()
            return
        blk = Block(name, to_meters(xv, unit), to_meters(yv, unit), to_meters(zv, unit))
        place_new_block_next_to_others(blk, self.blocks)
        self.blocks.append(blk)
        self.ensure_visible(blk)
        self.selected_id = blk.id

        self.add_name_var.set("")
        self.add_x_var.set("")
        self.add_y_var.set("")
        self.add_z_var.set("")
        self._add_name_entry.config(fg=TEXT_FAINT)
        self._add_name_entry.insert(0, "Name")

        self.refresh_panel()
        self.render()

    def _flash_form_error(self):
        orig = self._add_name_entry.cget("highlightbackground")
        self._add_name_entry.config(highlightbackground=DANGER)
        self.root.after(500, lambda: self._add_name_entry.config(highlightbackground=orig))

    def reset_layout(self):
        arrange_side_by_side(self.blocks)
        self.fit_camera_to_scene()
        self.refresh_panel()
        self.render()

    def toggle_units(self):
        self.unit_mode = "ft" if self.unit_mode == "m" else "m"
        self.units_btn.config(text="Units: " + ("m" if self.unit_mode == "m" else "ft\u00b7in"))
        self.refresh_panel()
        self.render()

    # ================================================================
    # panel rendering
    # ================================================================
    def refresh_panel(self):
        self._refresh_list()
        self._refresh_detail()

    def _refresh_list(self):
        for child in self.list_frame.winfo_children():
            child.destroy()
        if not self.blocks:
            tk.Label(self.list_frame, text="Nothing to compare yet.", bg=BG_PANEL, fg=TEXT_FAINT,
                     font=FONT_UI_SM).pack(pady=8)
            return
        for i, b in enumerate(self.blocks):
            selected = b.id == self.selected_id
            row = tk.Frame(self.list_frame, bg=BG_ELEVATED_HOVER if selected else BG_ELEVATED,
                            highlightthickness=1,
                            highlightbackground=ACCENT if selected else BG_ELEVATED, cursor="hand2")
            row.pack(fill="x", pady=3)

            swatch = tk.Frame(row, bg=b.color, width=11, height=11)
            swatch.pack(side="left", padx=(8, 8), pady=8)

            text_col = tk.Frame(row, bg=row["bg"])
            text_col.pack(side="left", fill="x", expand=True, pady=6)
            name_lbl = tk.Label(text_col, text=f"{i + 1}. {b.name}", bg=row["bg"], fg=TEXT,
                                 font=FONT_UI_SM, anchor="w")
            name_lbl.pack(fill="x")
            dims_lbl = tk.Label(text_col, text=b.dims_label(self.unit_mode), bg=row["bg"], fg=TEXT_DIM,
                                 font=FONT_MONO_SM, anchor="w")
            dims_lbl.pack(fill="x")

            del_btn = tk.Label(row, text="\u00d7", bg=row["bg"], fg=TEXT_FAINT, font=("Segoe UI", 13), cursor="hand2")
            del_btn.pack(side="right", padx=(0, 10))

            for widget in (row, swatch, text_col, name_lbl, dims_lbl):
                widget.bind("<Button-1>", lambda e, bid=b.id: self.select_block(bid))
            del_btn.bind("<Button-1>", lambda e, bid=b.id: self.delete_block(bid))
            del_btn.bind("<Enter>", lambda e, w=del_btn: w.config(fg=DANGER_STRONG))
            del_btn.bind("<Leave>", lambda e, w=del_btn: w.config(fg=TEXT_FAINT))

    def _refresh_detail(self):
        for child in self.detail_frame.winfo_children():
            child.destroy()
        b = self.find_block(self.selected_id)
        if not b:
            return

        tk.Label(self.detail_frame, text="SELECTED OBJECT", bg=BG_PANEL, fg=TEXT_FAINT,
                 font=FONT_LABEL_HEAD).pack(anchor="w", pady=(0, 6))
        card = tk.Frame(self.detail_frame, bg=BG_ELEVATED, highlightthickness=1, highlightbackground=LINE_SOFT)
        card.pack(fill="x")
        inner = tk.Frame(card, bg=BG_ELEVATED)
        inner.pack(fill="x", padx=12, pady=12)

        tk.Label(inner, text=b.name, bg=BG_ELEVATED, fg=TEXT, font=FONT_UI_SM, anchor="w").pack(fill="x")
        tk.Label(inner, text=b.dims_label(self.unit_mode), bg=BG_ELEVATED, fg=TEXT_DIM, font=FONT_MONO_SM,
                 anchor="w").pack(fill="x", pady=(0, 8))

        self.pos_vars = {}
        for axis in ("X", "Y", "Z"):
            row = tk.Frame(inner, bg=BG_ELEVATED)
            row.pack(fill="x", pady=2)
            tk.Label(row, text=axis, bg=BG_ELEVATED, fg=TEXT_FAINT, font=FONT_MONO, width=2).pack(side="left")
            var = tk.StringVar(value=f"{getattr(b, axis.lower()):.3f}")
            ent = tk.Entry(row, textvariable=var, bg=BG, fg=TEXT, insertbackground=TEXT, relief="flat",
                           font=FONT_MONO, highlightthickness=1, highlightbackground=LINE, highlightcolor=ACCENT)
            ent.pack(side="left", fill="x", expand=True)
            self.pos_vars[axis] = var
            var.trace_add("write", lambda *_a, axis=axis, var=var: self._on_pos_var_change(axis, var))
        tk.Label(inner, text="position in meters, from center", bg=BG_ELEVATED, fg=TEXT_FAINT,
                 font=FONT_MONO_SM).pack(anchor="w", pady=(6, 0))

    def _on_pos_var_change(self, axis, var):
        if getattr(self, "_suppress_pos_trace", False):
            return
        b = self.find_block(self.selected_id)
        if not b:
            return
        try:
            v = float(var.get())
        except ValueError:
            return
        if axis == "Z":
            v = max(0.0, v)
        setattr(b, axis.lower(), v)
        self.render()

    def update_detail_fields_live(self):
        b = self.find_block(self.selected_id)
        if not b or not hasattr(self, "pos_vars"):
            return
        # Suppress the write-trace while programmatically refreshing the
        # display during a drag -- otherwise setting the box to its
        # rounded-for-display text (e.g. "-0.003") re-parses and writes
        # that rounded value back over the drag's full-precision result,
        # on every axis, even the ones this drag never touched.
        self._suppress_pos_trace = True
        for axis in ("X", "Y", "Z"):
            self.pos_vars[axis].set(f"{getattr(b, axis.lower()):.3f}")
        self._suppress_pos_trace = False


def run(blocks):
    root = tk.Tk()
    root.title("Compare Objects")
    root.geometry("1280x860")
    root.minsize(820, 560)
    App(root, blocks)
    root.mainloop()


# ====================================================================
# Default objects — starts empty; add objects from the window itself
# ("Add object" form in the panel), or list some here to have them
# pre-loaded every time you run the script. Keep objects within a
# couple orders of magnitude of each other for a readable view --
# comparing a coin to a skyscraper works mathematically, but the coin
# will be a barely-visible sliver, same as on any real size-comparison
# site.
#
# Obj("Door", 0.81, 0.05, 2.03, "m"),
# Obj("Refrigerator", 0.90, 0.70, 1.80, "m"),
# ====================================================================

def Obj(name, x, y, z, unit="m"):
    return (name, x, y, z, unit)


OBJECTS = []


def main():
    Block.reset_color_cursor()
    blocks = []
    for name, x, y, z, unit in OBJECTS:
        blocks.append(Block(name, to_meters(x, unit), to_meters(y, unit), to_meters(z, unit)))
    arrange_side_by_side(blocks)
    run(blocks)


if __name__ == "__main__":
    main()
