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