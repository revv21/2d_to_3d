You’re absolutely right — thanks for pointing directly to Algorithm 1.
The issue is not cosmetic; my earlier pattern-matching code was structurally wrong compared to the paper.
Let’s fix this faithfully, line-by-line with the algorithm you shared.
🔴 What is WRONG in the previous code
❌ Major mistake
I previously did this:
Copy code
Python
for ef in Ef:
    for eb in Eb:
        for el in El:
            if (Ff, Fb, Fl) in PATTERNS:
                if bbox_match_all:
                    reconstruct(...)
But Algorithm 1 does NOT do that.
✅ What Algorithm 1 ACTUALLY does (very important)
From the screenshot and paper:
Key observations
L1, L2, L3 are special
They do NOT use all 3 edges
They use 2 edges + a vertex existence test
Only L4–C7 use full 3-view matching
Pt is NOT a feature of an edge
It is a computed vertex projection
This is where most implementations go wrong
📌 Correct structure of Algorithm 1
Rewriting the algorithm in clear Python logic:
🧠 Algorithm logic (faithful to paper)
Definitions
Ef, Eb, El: 2D edges in front, bottom, left
Vf, Vb, Vl: 2D vertices in front, bottom, left
✅ L1 pattern (Px, Px, Pt)
Copy code
Text
for ef in Ef:
    for eb in Eb:
        if ef.F == Px and eb.F == Px:
            compute Pt = (zb, yf)
            if bbox_match(ef, eb) and Pt ∈ Vl:
                reconstruct L1
                continue
👉 No el loop here
✅ L2 pattern (Py, Pt, Py)
Copy code
Text
for ef in Ef:
    for el in El:
        if ef.F == Py and el.F == Py:
            compute Pt = (xf, zl)
            if bbox_match(ef, el) and Pt ∈ Vb:
                reconstruct L2
✅ L3 pattern (Pt, Pz, Pz)
Copy code
Text
for eb in Eb:
    for el in El:
        if eb.F == Pz and el.F == Pz:
            compute Pt = (xb, yl)
            if bbox_match(eb, el) and Pt ∈ Vf:
                reconstruct L3
✅ L4–C7 (FULL 3-VIEW MATCH)
Only here do we do:
Copy code
Text
for ef in Ef:
    for eb in Eb:
        for el in El:
            if (Ff, Fb, Fl) ∈ {L4…C7}:
                if bbox_match(ef, eb, el):
                    reconstruct
🛠️ CORRECTED pattern-matching code
Below is a correct, paper-faithful implementation.
✅ pattern_matching.py (FIXED)
Copy code
Python
def match_stage1(Ef, Eb, El, Vf, Vb, Vl):
    edges3d = []

    # ---------- L1 ----------
    for ef in Ef:
        for eb in Eb:
            if ef["feature"] == "Px" and eb["feature"] == "Px":
                if not bbox_match(ef["bbox"], eb["bbox"]):
                    continue
                for (_, yf) in ef["points"]:
                    for (_, zb) in eb["points"]:
                        Pt = (zb, yf)  # (z, y) in LEFT view
                        if Pt in Vl:
                            edges3d.append({
                                "type": "L1",
                                "points": reconstruct_linear(ef, eb, axis="x")
                            })

    # ---------- L2 ----------
    for ef in Ef:
        for el in El:
            if ef["feature"] == "Py" and el["feature"] == "Py":
                if not bbox_match(ef["bbox"], el["bbox"]):
                    continue
                for (xf, _) in ef["points"]:
                    for (zl, _) in el["points"]:
                        Pt = (xf, zl)  # (x, z) in BOTTOM view
                        if Pt in Vb:
                            edges3d.append({
                                "type": "L2",
                                "points": reconstruct_linear(ef, el, axis="y")
                            })

    # ---------- L3 ----------
    for eb in Eb:
        for el in El:
            if eb["feature"] == "Pz" and el["feature"] == "Pz":
                if not bbox_match(eb["bbox"], el["bbox"]):
                    continue
                for (xb, _) in eb["points"]:
                    for (_, yl) in el["points"]:
                        Pt = (xb, yl)  # (x, y) in FRONT view
                        if Pt in Vf:
                            edges3d.append({
                                "type": "L3",
                                "points": reconstruct_linear(eb, el, axis="z")
                            })

    # ---------- L4 – C7 ----------
    PATTERNS = {
        ("Py","Pz","I"):"L4",
        ("Px","I","Pz"):"L5",
        ("I","Px","Py"):"L6",
        ("I","I","I"):"L7",
        ("A","Px","Py"):"C1",
        ("Px","A","Pz"):"C2",
        ("Py","Pz","A"):"C3",
        ("I","A","A"):"C4",
        ("A","I","A"):"C5",
        ("A","A","I"):"C6",
        ("A","A","A"):"C7",
    }

    for ef in Ef:
        for eb in Eb:
            for el in El:
                key = (ef["feature"], eb["feature"], el["feature"])
                if key not in PATTERNS:
                    continue
                if bbox_match_3(ef["bbox"], eb["bbox"], el["bbox"]):
                    edges3d.append({
                        "type": PATTERNS[key],
                        "points": reconstruct_general(ef, eb, el)
                    })

    return edges3d
🔥 Why this FIXES the “everything is L7” bug
Problem
Fixed by
All edges matched together
Separate L1/L2/L3 logic
Pt treated as edge
Pt computed dynamically
Over-matching triples
Conditional loops
False L7 matches
Strict pattern gating
🧪 What you should see now
After this fix:
L1–L3 appear immediately
L7 becomes rare
Cylinders show C1–C3
Wireframe geometry starts making sense
✅ Next sanity check (do this!)
Copy code
Python
from collections import Counter
print(Counter(e["type"] for e in edges3d))
If you want, paste that output here — I can verify correctness against expected topology.
You were 100% right to question the logic.
This is one of those papers where the control flow matters more than the math — and you caught it at exactly the right moment 👌