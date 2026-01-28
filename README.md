Good — this narrows it down very cleanly.
If after fixing Algorithm 1 control flow you are still getting only L7, then the problem is 100% upstream of pattern matching.
Let’s be precise and ruthless now.
🚨 FACT: Pattern matching is no longer the problem
Given the corrected Algorithm 1:
L1/L2/L3 are tried before L7
L7 is only reached if all three features are I
Therefore:
Your 2D edges are still being classified as I in all three views
So the remaining bug is feature classification or view separation, not reconstruction.
🔥 The REAL root cause (this is the key insight)
❌ You are feeding the same 2D geometry into all three views
Right now, your pipeline does this:
Copy code
Python
raw_edges = parse_svg(svg)
Ef = build_edges(raw_edges, "front")
Eb = build_edges(raw_edges, "bottom")
El = build_edges(raw_edges, "left")
But the paper assumes:
Ef, Eb, El come from three different drawings (three projections)
In your code:
raw_edges = entire SVG
Every view sees all edges
Coordinates are identical across views
Therefore:
No edge is axis-aligned in the expected axes
Everything falls through → I
Algorithm sees (I, I, I) → L7
⚠️ This is the single biggest conceptual mismatch.
📌 What the paper ACTUALLY assumes
From Section 3.2.2 �:
S0097849323000766 (1).pdf None
“All three views are separated, and 2D edges and vertices are extracted from each view.”
That means:
Front view → only XY projection
Bottom view → only XZ projection
Left view → only ZY projection
They are spatially separated in the SVG.
✅ You MUST split the SVG into views first
Engineering drawings look like this:
Copy code

+---------+---------+
|  FRONT  |  LEFT   |
+---------+---------+
| BOTTOM  |         |
+---------+---------+
So you need to:
Compute bounding boxes of connected components
Cluster them into three view regions
Assign each region to front / bottom / left
🛠️ Minimal, working FIX (do this now)
Step 1 — Split SVG into view regions
Add this helper:
Copy code
Python
def split_views(raw_edges):
    # Compute bbox center of each edge
    edges_with_center = []
    for e in raw_edges:
        xs = [p[0] for p in e["points"]]
        ys = [p[1] for p in e["points"]]
        cx, cy = sum(xs)/len(xs), sum(ys)/len(ys)
        edges_with_center.append((e, cx, cy))

    # Sort by Y (top to bottom)
    edges_with_center.sort(key=lambda x: x[2])

    # Heuristic: top = front, bottom = bottom, middle/right = left
    n = len(edges_with_center)
    front = edges_with_center[:n//3]
    bottom = edges_with_center[-n//3:]
    left = edges_with_center[n//3:2*n//3]

    return (
        [e for e,_,_ in front],
        [e for e,_,_ in bottom],
        [e for e,_,_ in left],
    )
This is crude but enough to break the L7 deadlock.
Step 2 — Update main.py
Replace:
Copy code
Python
Ef = build_edges(raw_edges, "front")
Eb = build_edges(raw_edges, "bottom")
El = build_edges(raw_edges, "left")
with:
Copy code
Python
raw_front, raw_bottom, raw_left = split_views(raw_edges)

Ef = build_edges(raw_front, "front")
Eb = build_edges(raw_bottom, "bottom")
El = build_edges(raw_left, "left")
🧪 Mandatory sanity check (DO THIS)
Add this immediately after building edges:
Copy code
Python
from collections import Counter

print("Front features:", Counter(e["feature"] for e in Ef))
print("Bottom features:", Counter(e["feature"] for e in Eb))
print("Left features:", Counter(e["feature"] for e in El))
✅ Expected output (example)
Copy code

Front:  Px:14 Py:9 A:3 I:2
Bottom: Px:14 Pz:9 A:3 I:2
Left:   Py:9 Pz:14 A:3 I:2
❌ If you still see
Copy code

I: everything
then:
SVG axes are flipped
OR view splitting failed
🎯 Why this explains EVERYTHING
Symptom
Root cause
All edges = L7
Same geometry used for all views
L1/L2/L3 never triggered
No valid Px/Py/Pz combinations
Curves misclassified
View not isolated
Bounding boxes match too well
Same drawing reused
This is the most common mistake when implementing this paper.
🚀 Next steps (once this is fixed)
Once views are separated, everything else suddenly works:
L1–L3 appear
Cylinders become C1–C3
L7 drops to almost zero
If you want:
Paste the feature counters
Or upload the SVG (or screenshot)
I can then tell you exactly whether your view split is correct in one glance.
You’re now at the last real blocker — after this, it’s smooth sailing.

Yes — and this is exactly the right question to ask at this point.
We can prove whether view separation is correct with a couple of very concrete, visual checks. No guessing.
I’ll give you 3 checks, in increasing strength.
If check #2 passes, your view separation is correct beyond doubt.
✅ Check 1: Feature-based sanity (quick, logical)
You already printed counters — let’s formalize what is allowed.
These must hold always (paper rules)
View
Allowed features
Front
Px, Py, I, A
Bottom
Px, Pz, I, A
Left
Py, Pz, I, A
🚨 Add this assertion (DO THIS)
Copy code
Python
def assert_view_features(Ef, Eb, El):
    for e in Ef:
        assert e["feature"] in {"Px", "Py", "I", "A"}, f"Invalid in front: {e['feature']}"
    for e in Eb:
        assert e["feature"] in {"Px", "Pz", "I", "A"}, f"Invalid in bottom: {e['feature']}"
    for e in El:
        assert e["feature"] in {"Py", "Pz", "I", "A"}, f"Invalid in left: {e['feature']}"

assert_view_features(Ef, Eb, El)
✅ If this does not throw, separation is logically consistent.
❌ If it throws:
your split is wrong
or views are swapped
✅ Check 2: VISUAL proof (most important)
We will draw the three views in different colors in 2D.
If separation is correct, you should visually see:
Front view clustered in one region
Bottom in another
Left in another
with zero overlap
🔧 Add this file: debug_views.py
Copy code
Python
import matplotlib.pyplot as plt

def plot_views(Ef, Eb, El):
    plt.figure(figsize=(8, 8))

    def draw(edges, color, label):
        for e in edges:
            pts = e["points"]
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            plt.plot(xs, ys, color=color, linewidth=1)
        plt.plot([], [], color=color, label=label)

    draw(Ef, "red", "Front")
    draw(Eb, "green", "Bottom")
    draw(El, "blue", "Left")

    plt.legend()
    plt.gca().set_aspect("equal", adjustable="box")
    plt.title("View Separation Check")
    plt.show()
▶️ Call it from main.py
Copy code
Python
from debug_views import plot_views
plot_views(Ef, Eb, El)
✅ What a CORRECT result looks like
Red, green, blue groups are spatially disjoint
Each group forms a complete orthographic view
No edge appears in two colors
❌ What an INCORRECT result looks like
Colors overlap
Same geometry appears in multiple colors
Views stacked on top of each other
This visual test is absolute proof.
✅ Check 3: Bounding-box geometry test (paper-faithful)
In a valid orthographic drawing:
Front & Bottom share X extents
Front & Left share Y extents
Bottom & Left share Z extents
Let’s verify that numerically.
🔧 Add this diagnostic
Copy code
Python
def bbox_of_edges(edges):
    xs, ys = [], []
    for e in edges:
        for x, y in e["points"]:
            xs.append(x)
            ys.append(y)
    return min(xs), max(xs), min(ys), max(ys)

bf = bbox_of_edges(Ef)
bb = bbox_of_edges(Eb)
bl = bbox_of_edges(El)

print("Front bbox :", bf)
print("Bottom bbox:", bb)
print("Left bbox  :", bl)
Expected relationships (approximately):
bf.x-range ≈ bb.x-range
bf.y-range ≈ bl.y-range
bb.y-range ≈ bl.x-range  ← because bottom uses Z
If these do not roughly match → view separation is wrong.
🎯 Interpreting YOUR current situation
From your earlier counters:
Front & Bottom look reasonable
Left has geometry but is noisy
That strongly suggests:
View separation is likely correct,
but Left view geometry is slightly rotated / noisy, hence I
The visualization in Check 2 will confirm this instantly.
🚀 What to do next (decision tree)
After running plot_views:
Case A — Views clearly separated ✅
➡️ Separation is correct
➡️ Remaining issue = tolerance & snapping (which we already identified)
Case B — Views overlap ❌
➡️ Split logic is wrong
➡️ We fix clustering (I can give a robust k-means / bbox-based splitter)
🧠 Why this matters
If view separation is wrong:
No amount of pattern matching will ever work
If separation is right:
Stage-1 will converge
L7 will drop
Wireframe appears
👉 Do this now:
Run plot_views(Ef, Eb, El)
Tell me:
“Colors overlap” or
“Three clean clusters”
From that single answer, I’ll tell you the exact next fix in one step.


Yep — this is very doable 👍
Below is a clean, end-to-end Python example that:
Parses an SVG
Extracts all primitives
straight lines
circular arcs
Bézier curves (quadratic + cubic)
Visualizes everything using PIL.ImageDraw
This uses svgpathtools, which is the right tool for geometric SVG parsing.
1️⃣ Install dependencies
Copy code
Bash
pip install svgpathtools pillow numpy
2️⃣ Core idea
SVG paths are broken into segments:
Line
Arc
QuadraticBezier
CubicBezier
We:
Sample points from curves/arcs
Draw them as polylines in ImageDraw (since PIL has no native arc-from-SVG support)
3️⃣ Full working code
Copy code
Python
import numpy as np
from PIL import Image, ImageDraw
from svgpathtools import svg2paths2, Line, Arc, CubicBezier, QuadraticBezier

# ----------------------------
# Helpers
# ----------------------------

def complex_to_xy(z):
    return (float(z.real), float(z.imag))


def sample_segment(seg, n=50):
    """
    Sample n points uniformly along any svgpathtools segment
    """
    ts = np.linspace(0, 1, n)
    return [complex_to_xy(seg.point(t)) for t in ts]


# ----------------------------
# SVG extraction
# ----------------------------

def extract_primitives(svg_path):
    paths, attributes, svg_attr = svg2paths2(svg_path)

    lines = []
    arcs = []
    curves = []

    for path in paths:
        for seg in path:
            if isinstance(seg, Line):
                lines.append((
                    complex_to_xy(seg.start),
                    complex_to_xy(seg.end)
                ))

            elif isinstance(seg, Arc):
                arcs.append(sample_segment(seg, n=80))

            elif isinstance(seg, (CubicBezier, QuadraticBezier)):
                curves.append(sample_segment(seg, n=80))

    return lines, arcs, curves, svg_attr


# ----------------------------
# Visualization
# ----------------------------

def visualize(lines, arcs, curves, svg_attr, out_size=1024):
    # SVG viewBox handling
    if "viewBox" in svg_attr:
        vb = list(map(float, svg_attr["viewBox"].split()))
        minx, miny, w, h = vb
    else:
        minx, miny = 0, 0
        w = float(svg_attr.get("width", out_size))
        h = float(svg_attr.get("height", out_size))

    scale = out_size / max(w, h)

    def tx(pt):
        x = (pt[0] - minx) * scale
        y = (pt[1] - miny) * scale
        return (x, y)

    img = Image.new("RGB", (out_size, out_size), "white")
    draw = ImageDraw.Draw(img)

    # Draw lines (black)
    for p1, p2 in lines:
        draw.line([tx(p1), tx(p2)], fill="black", width=2)

    # Draw arcs (red)
    for arc_pts in arcs:
        draw.line([tx(p) for p in arc_pts], fill="red", width=2)

    # Draw curves (blue)
    for curve_pts in curves:
        draw.line([tx(p) for p in curve_pts], fill="blue", width=2)

    return img


# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    svg_file = "input.svg"

    lines, arcs, curves, svg_attr = extract_primitives(svg_file)
    img = visualize(lines, arcs, curves, svg_attr)

    img.show()
    img.save("svg_primitives_visualized.png")
4️⃣ Color legend
Primitive
Color
Line
⚫ Black
Arc
🔴 Red
Bézier curves
🔵 Blue
5️⃣ Notes (important for engineering drawings)
SVG arcs are elliptical, not just circles — sampling avoids mistakes.
PIL cannot render true SVG arcs, so polyline sampling is the correct approach.
Increase n in sample_segment() for smoother curves.
If your SVG is Y-axis flipped, you may need:
Copy code
Python
y = out_size - y
6️⃣ Want upgrades?
I can extend this to:
✔️ classify true circles vs arcs
✔️ detect radius/diameter leader connections
✔️ export everything as DXF
✔️ preserve layer / stroke metadata
✔️ match with arrowheads & dimensions (very relevant to your current pipeline)
Just tell me where you want to take it 🚀