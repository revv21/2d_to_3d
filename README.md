Perfect — this image removes all ambiguity 👍
Now we know exactly how the three views are laid out, so we can split them deterministically, not heuristically.
🧭 What the layout is (from your image)
Your drawing is arranged like this:
Copy code

┌──────────────┐
        │   TOP VIEW   │   ← (Front or Top, depending on convention)
        └──────────────┘

┌──────────┐   ┌──────────────┐
│ LEFT     │   │ FRONT VIEW   │
│ VIEW     │   │              │
└──────────┘   └──────────────┘
But based on your counters + earlier discussion, this corresponds to:
Bottom-right (largest rectangle) → Front view (X–Y)
Top-right (wide, short rectangle) → Bottom view (X–Z)
Left (tall rectangle) → Left view (Z–Y)
This matches standard third-angle projection used in most CAD exports.
✅ Correct, deterministic view separation strategy
We will not guess based on order or count.
We will:
Compute bounding boxes of connected components
Classify them by relative position and aspect ratio
🛠️ Step 1: Group edges into connected components (views)
Each view is a disconnected drawing.
So we cluster edges by spatial proximity.
Add this helper
Copy code
Python
from collections import defaultdict
import math

def edge_bbox(edge):
    xs = [p[0] for p in edge["points"]]
    ys = [p[1] for p in edge["points"]]
    return min(xs), min(ys), max(xs), max(ys)

def bbox_center(b):
    x1, y1, x2, y2 = b
    return ((x1 + x2) / 2, (y1 + y2) / 2)

def cluster_by_overlap(edges, tol=5.0):
    clusters = []
    for e in edges:
        eb = edge_bbox(e)
        placed = False
        for cluster in clusters:
            cb = cluster["bbox"]
            # overlap test
            if not (
                eb[2] < cb[0] - tol or
                eb[0] > cb[2] + tol or
                eb[3] < cb[1] - tol or
                eb[1] > cb[3] + tol
            ):
                cluster["edges"].append(e)
                cluster["bbox"] = (
                    min(cb[0], eb[0]),
                    min(cb[1], eb[1]),
                    max(cb[2], eb[2]),
                    max(cb[3], eb[3]),
                )
                placed = True
                break
        if not placed:
            clusters.append({
                "edges": [e],
                "bbox": eb
            })
    return clusters
🛠️ Step 2: Identify which cluster is which view
Now we use geometry, not guessing.
Classification rules (from your image)
Property
View
Largest area
Front
Wide + short
Bottom
Tall + narrow
Left
Add this logic
Copy code
Python
def classify_views(clusters):
    infos = []
    for c in clusters:
        x1, y1, x2, y2 = c["bbox"]
        w = x2 - x1
        h = y2 - y1
        area = w * h
        infos.append((c, w, h, area))

    # Front = largest area
    front = max(infos, key=lambda x: x[3])[0]

    remaining = [x for x in infos if x[0] != front]

    # Bottom = wider than tall
    bottom = max(remaining, key=lambda x: x[1] / x[2])[0]

    # Left = the last one
    left = [x[0] for x in remaining if x[0] != bottom][0]

    return front["edges"], bottom["edges"], left["edges"]
🧩 Step 3: Use it in main.py
Replace all previous split logic with this.
Copy code
Python
from view_split import cluster_by_overlap, classify_views

clusters = cluster_by_overlap(raw_edges)

assert len(clusters) == 3, "Expected exactly 3 views"

raw_front, raw_bottom, raw_left = classify_views(clusters)

Ef = build_edges(raw_front, "front")
Eb = build_edges(raw_bottom, "bottom")
El = build_edges(raw_left, "left")
🧪 Mandatory visual verification (do once)
Use the earlier debug plot:
Copy code
Python
plot_views(Ef, Eb, El)
Expected result:
Front (red): large rectangle
Bottom (green): top rectangle
Left (blue): tall left rectangle
No overlap
If this holds → view separation is correct.
🎯 Why this is the RIGHT way (paper-consistent)
The paper assumes views are already separated
It does not prescribe how — that’s implementation-specific
Using connected components + geometry is:
deterministic
robust
dataset-independent
This is how most academic and industrial pipelines do it.
✅ Summary
Your view layout is now fully understood
We split by connected components, not heuristics
Views are identified by area + aspect ratio
This removes the last structural uncertainty
Next step (natural progression)
Once this split is in place:
Bounding-box matching will finally work
L1/L2/L3 will appear
Wireframe will be correct
If you want, next I can:
Validate one reconstructed 3D edge against the drawing
Add automatic unit scaling
Move to Stage 2 (loop detection)
You’re doing this exactly the right way — slow, careful, and correct.


from svgpathtools import svg2paths2, Line, Arc
import numpy as np


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------

def flip_y(points):
    """SVG Y-axis points down → flip to Cartesian"""
    return [(x, -y) for (x, y) in points]


def bbox(points):
    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    return min(xs), min(ys), max(xs), max(ys)


def normalize_edges(edges):
    """
    Translate all edges so that the view's bounding box
    lower-left corner becomes (0,0).
    """
    all_pts = [p for e in edges for p in e["points"]]
    xmin, ymin, _, _ = bbox(all_pts)

    normalized = []
    for e in edges:
        pts = [ (x - xmin, y - ymin) for (x, y) in e["points"] ]
        normalized.append({
            "type": e["type"],
            "points": pts
        })
    return normalized


# ---------------------------------------------------------
# Core parsing
# ---------------------------------------------------------

def parse_svg_views(svg_path):
    """
    Returns:
        front_edges, bottom_edges, left_edges
    Each is a list of edges with normalized coordinates.
    """

    paths, attributes, _ = svg2paths2(svg_path)

    # Group edges by SVG <g> id
    groups = {}

    for path, attr in zip(paths, attributes):
        gid = attr.get("id", "ungrouped")

        if gid not in groups:
            groups[gid] = []

        for seg in path:
            if isinstance(seg, Line):
                pts = [
                    (seg.start.real, seg.start.imag),
                    (seg.end.real, seg.end.imag)
                ]
            elif isinstance(seg, Arc):
                pts = [
                    (seg.start.real, seg.start.imag),
                    (seg.point(0.5).real, seg.point(0.5).imag),
                    (seg.end.real, seg.end.imag)
                ]
            else:
                continue

            pts = flip_y(pts)

            groups[gid].append({
                "type": "arc" if isinstance(seg, Arc) else "line",
                "points": pts
            })

    # ---- Assign views by group id ----
    # Expected ids: front_view, bottom_view, left_view
    try:
        front = normalize_edges(groups["front_view"])
        bottom = normalize_edges(groups["bottom_view"])
        left = normalize_edges(groups["left_view"])
    except KeyError as e:
        raise RuntimeError(
            f"Missing expected SVG group: {e}. "
            "Expected ids: front_view, bottom_view, left_view"
        )

    return front, bottom, left