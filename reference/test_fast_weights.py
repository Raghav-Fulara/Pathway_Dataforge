"""
test_fast_weights.py — the honesty check.

Runs two things:
  1. Property tests of the NumPy reference implementation (fast_weights.py):
     perfect recall at load 1, monotone-ish degradation, closed-form identity,
     decay behavior.
  2. Cross-implementation equivalence: the JSON exported by the NumPy reference
     vs js_vectors.json produced by running the ACTUAL <script> core extracted
     from index.html (see extract_and_run_core.js). Same seeds -> same numbers
     within 1e-9. This is the check that what the learner sees in the browser is
     the same math as the auditable Python reference — i.e., the artifact is
     live computation, not an animation.

Usage:
    python3 reference/fast_weights.py                 # exports expected_vectors.json
    node reference/extract_and_run_core.js            # exports js_vectors.json
    pytest reference/test_fast_weights.py -q   (or: python3 reference/test_fast_weights.py)
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fast_weights import (capacity_curve, decay_curve, export_vectors,
                          full_decompose, hebbian_write, make_pair_generator,
                          make_state, read, cosine)

TOL = 1e-9


def test_recall_perfect_at_load_one():
    rows = capacity_curve(n_pairs=3)
    assert rows[0]["meanSim"] > 0.9999
    assert rows[0]["exactRate"] == 1.0
    assert rows[0]["interferenceShare"] < 1e-9


def test_interference_grows_and_recall_degrades():
    rows = capacity_curve(n_pairs=40)
    assert rows[-1]["interferenceShare"] > rows[3]["interferenceShare"]
    assert rows[-1]["exactRate"] < 1.0
    assert rows[-1]["meanSim"] < rows[3]["meanSim"]


def test_closed_form_identity():
    d, lam, seed = 24, 0.9, 5
    gen = make_pair_generator(d, 0.2, 0.0, seed)
    M = make_state(d)
    ks, vs = [], []
    for _ in range(7):
        k, v = gen()
        M = hebbian_write(M, k, v, lam)
        ks.append(k); vs.append(v)
    q = ks[2]
    r, terms = full_decompose(ks, vs, lam, q)
    assert np.allclose(read(M, q), r, atol=1e-12)          # sum of terms == matrix read
    assert abs(terms[2][0] - lam**4 * float(ks[2] @ q)) < 1e-12  # target weight formula


def test_state_shape_never_grows():
    d = 16
    M = make_state(d)
    gen = make_pair_generator(d, 0.2, 0.0, 1)
    for _ in range(50):
        k, v = gen()
        M = hebbian_write(M, k, v, 0.99)
    assert M.shape == (d, d)  # 50 tokens stored, state still d x d


def test_decay_forgets_first_pair():
    rows = decay_curve(n_pairs=20, lam=0.85)
    assert rows[0]["simFirst"] > rows[-1]["simFirst"]
    assert rows[-1]["simFirst"] < 0.9


def _load(name):
    p = os.path.join(HERE, name)
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def test_js_core_matches_numpy():
    py = _load("expected_vectors.json")
    js = _load("js_vectors.json")
    if py is None:
        py_path = export_vectors(os.path.join(HERE, "expected_vectors.json"))
        py = _load("expected_vectors.json")
    assert js is not None, (
        "js_vectors.json missing — run: node reference/extract_and_run_core.js")
    assert py["params"] == js["params"]
    d = py["params"]["d"]
    # pairs and states
    for i in range(len(py["pairs"])):
        assert np.allclose(py["pairs"][i]["k"], js["pairs"][i]["k"], atol=TOL)
        assert np.allclose(py["pairs"][i]["v"], js["pairs"][i]["v"], atol=TOL)
        assert np.allclose(np.array(py["states"][i]), np.array(js["states"][i]).reshape(d, d), atol=TOL), f"state {i} differs"
    assert np.allclose(py["retrievalAtProbe"], js["retrievalAtProbe"], atol=TOL)
    # capacity + decay heads
    for a, b in zip(py["capacityHead"], js["capacityHead"]):
        assert abs(a["meanSim"] - b["meanSim"]) < TOL
        assert abs(a["exactRate"] - b["exactRate"]) < TOL
        assert abs(a["interferenceShare"] - b["interferenceShare"]) < TOL
    for a, b in zip(py["decayHead"], js["decayHead"]):
        assert abs(a["simFirst"] - b["simFirst"]) < TOL
    print("JS core == NumPy reference on fixed seeds (tol 1e-9). Artifact math verified.")


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS  {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} tests passed.")
