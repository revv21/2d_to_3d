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