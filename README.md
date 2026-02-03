import xml.etree.ElementTree as ET


# -----------------------------
# Helpers
# -----------------------------

def local_name(tag):
    """Strip namespace if present"""
    return tag.split('}', 1)[-1]


def bbox_from_points(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def normalize_edges(edges):
    """
    Translate edges so that the minimum (x,y) becomes (0,0)
    """
    all_pts = [p for e in edges for p in e["points"]]
    xmin, ymin, _, _ = bbox_from_points(all_pts)

    norm = []
    for e in edges:
        pts = [(x - xmin, y - ymin) for (x, y) in e["points"]]
        norm.append({
            "type": "line",
            "points": pts
        })
    return norm


# -----------------------------
# Core function
# -----------------------------

def parse_svg_views(svg_text):
    """
    Parse SVG text and return:
        front_edges, bottom_edges, left_edges
    Each edge = { "type": "line", "points": [(x1,y1),(x2,y2)] }
    """

    root = ET.fromstring(svg_text)

    groups = []

    # -----------------------------
    # Iterate over <g> elements
    # -----------------------------
    for g in root.iter():
        if local_name(g.tag) != "g":
            continue

        lines = []

        for child in g.iter():
            if local_name(child.tag) != "line":
                continue

            def get_float(attr):
                v = child.get(attr)
                return float(v) if v is not None else None

            x1 = get_float("x1")
            y1 = get_float("y1")
            x2 = get_float("x2")
            y2 = get_float("y2")

            if None in (x1, y1, x2, y2):
                continue
import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.svg import SVGBackend
from ezdxf.addons.drawing.config import Configuration


def dxf_to_svg(dxf_path, svg_path):
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    ctx = RenderContext(doc)
    ctx.set_current_layout(msp)

    backend = SVGBackend()

    config = Configuration()
    config.background = "#ffffff"       # background color
    config.foreground = "#000000"       # entity color
    config.lineweight_scaling = 1.0
    config.min_lineweight = 0.1

    frontend = Frontend(ctx, backend, config)
    frontend.draw_layout(msp, finalize=True)

    backend.save(svg_path)


if __name__ == "__main__":
    dxf_to_svg(
        r"C:\Users\inp_revanth\Downloads\X8X00614_all_vectors.dxf",
        "output.svg"
    )
            # Flip Y axis (SVG → Cartesian)
            y1 = -y1
            y2 = -y2

            lines.append({
                "type": "line",
                "points": [(x1, y1), (x2, y2)]
            })

        if not lines:
            continue

        # Compute group bounding box & centroid
        all_pts = [p for e in lines for p in e["points"]]
        xmin, ymin, xmax, ymax = bbox_from_points(all_pts)
        cx = 0.5 * (xmin + xmax)

        groups.append({
            "edges": lines,
            "bbox": (xmin, ymin, xmax, ymax),
            "cx": cx
        })

    # -----------------------------
    # Sanity check
    # -----------------------------
    if len(groups) != 3:
        raise RuntimeError(
            f"Expected exactly 3 view groups, found {len(groups)}"
        )

    # -----------------------------
    # Sort views left → right
    # -----------------------------
    groups.sort(key=lambda g: g["cx"])

    front  = normalize_edges(groups[0]["edges"])
    bottom = normalize_edges(groups[1]["edges"])
    left   = normalize_edges(groups[2]["edges"])

    return front, bottom, left


import ezdxf
from ezdxf.addons.drawing import Frontend, RenderContext
from ezdxf.addons.drawing.svg import SVGBackend
from ezdxf.addons.drawing.config import Configuration


def dxf_to_svg(dxf_path, svg_path, scale=1.0):
    # Load DXF
    doc = ezdxf.readfile(dxf_path)
    msp = doc.modelspace()

    # Rendering context
    ctx = RenderContext(doc)
    ctx.set_current_layout(msp)

    # SVG backend
    backend = SVGBackend()

    # Optional render config
    config = Configuration(
        background_policy=Configuration.BackgroundPolicy.WHITE,
        color_policy=Configuration.ColorPolicy.BLACK,
        lineweight_scaling=1.0,
    )

    # Render
    frontend = Frontend(ctx, backend, config)
    frontend.draw_layout(msp, finalize=True)

    # Save SVG
    backend.save(svg_path)


if __name__ == "__main__":
    dxf_to_svg("input.dxf", "output.svg")


Yep — this is a great question, and you’re reading the paper exactly the right way 👍
Below is a faithful, implementable version of Algorithm 1 (preprocessing + vectorization), and then a clean separation of lines vs curves from the obtained vectors, closely following the paper’s intent.
I’ll give you:
High-level pipeline (mapped exactly to Algorithm 1)
Concrete Python implementation (modular, readable)
How to classify vectors into straight lines vs curves
Notes on deviations / practical simplifications
Everything here is directly grounded in Algorithm 1 and Section 3.1 of the paper .
1️⃣ Mapping Algorithm 1 → Code
From Algorithm 1 (page 8):
Paper step
Implementation
Skeletonize + smoothing
skimage.morphology.skeletonize, Gaussian blur
Split skeleton to strokes
Graph traversal on skeleton pixels
Fit cubic Bézier
Least squares fitting
Sample n points
Uniform t ∈ [0,1]
Vector features
Geometry + curvature
Separate lines & curves
Curvature / deviation test
We’ll stop before graph construction, exactly as you asked.
2️⃣ Python Implementation (Preprocessing + Vectorization)
Dependencies
Copy code
Bash
pip install numpy opencv-python scikit-image scipy
A. Skeletonization + smoothing
Copy code
Python
import cv2
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import gaussian_filter
Copy code
Python
def skeletonize_image(img):
    """
    img: grayscale or binary image
    """
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(img, 200, 1, cv2.THRESH_BINARY_INV)
    skel = skeletonize(bw).astype(np.uint8)

    # slight smoothing (paper mentions this explicitly)
    skel = gaussian_filter(skel.astype(float), sigma=0.5) > 0
    return skel.astype(np.uint8)
B. Junction / endpoint detection (Fig. 4 in paper)
Copy code
Python
def neighbor_count(skel, y, x):
    h, w = skel.shape
    cnt = 0
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel[ny, nx]:
                cnt += 1
    return cnt


def classify_pixels(skel):
    endpoints = []
    junctions = []
    passings = []

    ys, xs = np.where(skel)
    for y, x in zip(ys, xs):
        n = neighbor_count(skel, y, x)
        if n == 1:
            endpoints.append((y, x))
        elif n >= 3:
            junctions.append((y, x))
        else:
            passings.append((y, x))

    return endpoints, junctions, passings
C. Stroke tracing (exactly Fig. 4(b,c))
Copy code
Python
def trace_strokes(skel):
    visited = set()
    strokes = []

    endpoints, junctions, _ = classify_pixels(skel)
    start_points = endpoints + junctions

    H, W = skel.shape

    def neighbors(p):
        y, x = p
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W and skel[ny, nx]:
                    yield (ny, nx)

    for sp in start_points:
        if sp in visited:
            continue

        for nb in neighbors(sp):
            if nb in visited:
                continue

            stroke = [sp, nb]
            visited.add(sp)
            visited.add(nb)

            prev = sp
            curr = nb

            while True:
                nbs = [p for p in neighbors(curr) if p != prev]
                if len(nbs) != 1:
                    break
                nxt = nbs[0]
                if nxt in visited:
                    break
                stroke.append(nxt)
                visited.add(nxt)
                prev, curr = curr, nxt

            if len(stroke) >= 4:
                strokes.append(np.array(stroke))

    return strokes
D. Split strokes at corners (θ second derivative)
Copy code
Python
def split_at_corners(stroke, angle_thresh=0.3):
    pts = stroke.astype(float)

    vecs = np.diff(pts, axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    vecs /= (norms + 1e-6)

    angles = np.arccos(np.clip(np.sum(vecs[:-1] * vecs[1:], axis=1), -1, 1))
    d2 = np.abs(np.diff(angles, n=2))

    split_idxs = np.where(d2 > angle_thresh)[0] + 2
    splits = np.split(pts, split_idxs)

    return [s for s in splits if len(s) >= 4]
E. Cubic Bézier fitting (least squares)
Copy code
Python
def fit_cubic_bezier(points):
    """
    Returns 4 control points
    """
    P = points
    n = len(P)

    t = np.linspace(0, 1, n)

    B = np.column_stack([
        (1 - t) ** 3,
        3 * (1 - t) ** 2 * t,
        3 * (1 - t) * t ** 2,
        t ** 3
    ])

    # fix endpoints
    C0 = P[0]
    C3 = P[-1]

    A = B[:, 1:3]
    Y = P - np.outer(B[:, 0], C0) - np.outer(B[:, 3], C3)

    C1C2, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)

    C1, C2 = C1C2
    return np.vstack([C0, C1, C2, C3])
F. Sample points from Bézier
Copy code
Python
def sample_bezier(ctrl, n=10):
    t = np.linspace(0, 1, n)
    B = (
        np.outer((1 - t) ** 3, ctrl[0]) +
        np.outer(3 * (1 - t) ** 2 * t, ctrl[1]) +
        np.outer(3 * (1 - t) * t ** 2, ctrl[2]) +
        np.outer(t ** 3, ctrl[3])
    )
    return B
3️⃣ Separating Lines vs Curves (Your Main Ask)
Key insight from the paper
Most vectors are straight lines; curves have non-zero curvature
A. Curvature computation
Copy code
Python
def curvature(points):
    d1 = np.gradient(points, axis=0)
    d2 = np.gradient(d1, axis=0)

    num = np.abs(d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0])
    denom = (np.linalg.norm(d1, axis=1) ** 3 + 1e-6)

    return num / denom
B. Line vs curve classifier
Copy code
Python
def classify_vector(sampled_pts, curvature_thresh=1e-2):
    k = curvature(sampled_pts)
    mean_k = np.mean(k)

    if mean_k < curvature_thresh:
        return "line"
    else:
        return "curve"
C. Full pipeline usage
Copy code
Python
img = cv2.imread("drawing.png", cv2.IMREAD_GRAYSCALE)
skel = skeletonize_image(img)

strokes = trace_strokes(skel)

vectors = []
for s in strokes:
    parts = split_at_corners(s)
    for p in parts:
        bez = fit_cubic_bezier(p)
        samples = sample_bezier(bez, n=10)
        kind = classify_vector(samples)
        vectors.append({
            "control_points": bez,
            "samples": samples,
            "type": kind
        })
4️⃣ What You Now Have
You now have:
✔ Vectorized drawing
✔ Cubic Bézier representation
✔ Uniform sampling
✔ Explicit separation of lines vs curves
✔ Ready input for feature extraction / graph construction
This is fully aligned with Algorithm 1 and Section 3.1 of the paper .
5️⃣ Important Practical Notes
The paper uses Datta thinning — skimage.skeletonize is a close approximation.
Corner splitting is crucial — skipping it causes curves + corners to merge.
Curvature threshold is data-dependent (tune once).
Circles & arcs → classified as curves (correct behavior).
Want next?
I can:
Add feature vector construction (Table 1)
Convert these vectors directly to DXF / SVG
Build graph edges exactly as EDGNet
Optimize this to run fast on large drawings
Just tell me what you want next 🔥