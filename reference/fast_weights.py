"""
fast_weights.py — NumPy reference implementation of the exact mathematics that runs
inside the interactive artifact (index.html, <script id="fw-core">).

Both implement the same update rules:

    Hebbian write:   M_t = lam * M_{t-1} + v k^T           (outer-product write)
    Read with cue q: r    = M_t q                          (associative retrieval)
    Closed form:     M_t q = sum_i lam^(t-i) (k_i . q) v_i (decay-weighted similarity sum)

The state M has fixed shape (d x d): no gradient step, no per-token memory slot.

All randomness flows through PortRng (a 32-bit LCG), which is trivially portable
to JavaScript, so fixed-seed experiments are reproducible across BOTH
implementations (see test_fast_weights.py — the honesty check).

License: MIT (see sources/licenses.md)
"""

import json
import math
import numpy as np


# ----------------------------------------------------------------------------
# Portable RNG (identical to the JS core; do not change one without the other)
# ----------------------------------------------------------------------------

class PortRng:
    """Deterministic LCG: s = (1664525*s + 1013904223) mod 2^32; u = s / 2^32."""

    def __init__(self, seed: int):
        self.s = seed & 0xFFFFFFFF
        if self.s == 0:
            self.s = 1

    def next(self) -> float:
        self.s = (1664525 * self.s + 1013904223) & 0xFFFFFFFF
        return self.s / 4294967296.0

    def choice(self, n: int, k: int):
        """k distinct indices from range(n) via partial Fisher-Yates (like JS)."""
        idx = list(range(n))
        for i in range(k):
            j = i + int(self.next() * (n - i))
            idx[i], idx[j] = idx[j], idx[i]
        return idx[:k]

    def uniform(self, n: int, lo: float, hi: float):
        return [lo + (hi - lo) * self.next() for _ in range(n)]


# ----------------------------------------------------------------------------
# Core ops (mirror of the JS core in index.html)
# ----------------------------------------------------------------------------

def make_state(d: int) -> np.ndarray:
    return np.zeros((d, d), dtype=np.float64)


def hebbian_write(M, k, v, lam=1.0):
    """M_t = lam * M + v k^T. lam=1 -> purely additive accumulation
    (the per-demonstration case; see BDH-CQ, arXiv:2608.09888)."""
    return lam * M + np.outer(v, k)


def read(M, q):
    """Associative read with cue q."""
    return M @ q


def decompose_read(keys, values, lam, q, t=None):
    """Closed form: r = target + crosstalk, where
       target    = lam^(t-1-i*) (k_i* . q) v_i*   for the matching key i*
       crosstalk = sum_{i != i*} lam^(t-1-i) (k_i . q) v_i
    Used by the artifact to visualise interference as a stacked bar."""
    t = len(keys) if t is None else t
    total = np.zeros_like(values[0])
    target = np.zeros_like(values[0])
    cross = np.zeros_like(values[0])
    for i in range(t):
        w = lam ** (t - 1 - i) * float(np.dot(keys[i], q))
        total = total + w * values[i]
    return total, target, cross


def full_decompose(keys, values, lam, q):
    """Per-term contributions [(weight, v_i)] so the UI can show the sum."""
    t = len(keys)
    terms = []
    for i in range(t):
        w = lam ** (t - 1 - i) * float(np.dot(keys[i], q))
        terms.append((w, values[i]))
    r = np.zeros_like(values[0])
    for w, v in terms:
        r = r + w * v
    return r, terms


# ----------------------------------------------------------------------------
# Pattern generation (sparse, non-negative — cf. BDH's sparse positive activations)
# ----------------------------------------------------------------------------

def _round_half_up(x: float) -> int:
    """Matches JS Math.floor(x + 0.5) exactly (Python round() is banker's rounding)."""
    return int(math.floor(x + 0.5))


def sparse_binary_pattern(rng: PortRng, d: int, sparsity: float) -> np.ndarray:
    x = np.zeros(d)
    n_on = max(1, _round_half_up(sparsity * d))
    for i in rng.choice(d, n_on):
        x[i] = 1.0
    return x


def make_pair_generator(d, sparsity, overlap, seed=42):
    """Stream of (key, value) associations with controllable overlap between
    consecutive keys. Overlap -> correlated keys -> crosstalk -> interference."""
    rng_k = PortRng(seed)
    rng_v = PortRng(seed + 1)
    state = {"prev": None}

    def next_pair():
        k = sparse_binary_pattern(rng_k, d, sparsity)
        prev = state["prev"]
        if prev is not None and overlap > 0:
            k_on = [i for i in range(d) if k[i] > 0]
            p_on = [i for i in range(d) if prev[i] > 0]
            n_swap = _round_half_up(overlap * min(len(k_on), len(p_on)))
            if n_swap > 0:
                for j in range(n_swap):
                    k[k_on[j]] = 0.0
                    k[p_on[j]] = 1.0
        v = sparse_binary_pattern(rng_v, d, sparsity)
        state["prev"] = k.copy()
        return k, v

    return next_pair


# ----------------------------------------------------------------------------
# Metrics and experiments (what the artifact plots live)
# ----------------------------------------------------------------------------

def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def exact_match(r, values):
    """Retrieved vector is 'exact' if the nearest stored value is the right one."""
    dists = [float(np.linalg.norm(r - v)) for v in values]
    return int(np.argmin(dists))


def capacity_curve(d=24, lam=1.0, n_pairs=40, sparsity=0.2, overlap=0.0, seed=42,
                   cue_noise=0.0, nonneg=False):
    """Store n_pairs associations one by one; after each write, probe recall of
    every stored pair. This is the artifact's live 'break the memory' experiment:
    recall is perfect at low load and degrades as interference accumulates."""
    rng_noise = PortRng(seed + 99)
    gen = make_pair_generator(d, sparsity, overlap, seed)
    M = make_state(d)
    keys, values, rows = [], [], []
    for t in range(1, n_pairs + 1):
        k, v = gen()
        M = hebbian_write(M, k, v, lam)
        keys.append(k)
        values.append(v)
        sims, exacts, shares = [], [], []
        for i in range(t):
            q = keys[i]
            if cue_noise > 0:
                q = q + np.array(rng_noise.uniform(d, -cue_noise, cue_noise))
            r = read(M, q)
            if nonneg:
                r = np.maximum(r, 0.0)
            sims.append(cosine(r, values[i]))
            exacts.append(1.0 if exact_match(r, values) == i else 0.0)
            target_w = lam ** (t - 1 - i) * float(np.dot(keys[i], q))
            target = target_w * values[i]
            shares.append(float(np.linalg.norm(r - target) /
                                (np.linalg.norm(r) + 1e-12)))
        rows.append({"load": t,
                     "meanSim": float(np.mean(sims)),
                     "exactRate": float(np.mean(exacts)),
                     "interferenceShare": float(np.mean(shares))})
    return rows


def decay_curve(d=24, sparsity=0.2, n_pairs=8, lam=0.9, seed=7):
    """Temporal forgetting: probe the FIRST pair while later pairs keep arriving."""
    gen = make_pair_generator(d, sparsity, 0.0, seed)
    M = make_state(d)
    k0, v0 = gen()
    M = hebbian_write(M, k0, v0, lam)
    rows = []
    for t in range(1, n_pairs + 1):
        k, v = gen()
        M = hebbian_write(M, k, v, lam)
        rows.append({"laterPairs": t, "simFirst": cosine(read(M, k0), v0)})
    return rows


# ----------------------------------------------------------------------------
# Export fixed-seed vectors for the JS-vs-NumPy honesty test
# ----------------------------------------------------------------------------

def export_vectors(path="expected_vectors.json"):
    d, lam, seed, sparsity, overlap = 24, 0.9, 1234, 0.2, 0.0
    gen = make_pair_generator(d, sparsity, overlap, seed)
    M = make_state(d)
    pairs, states = [], []
    for _ in range(6):
        k, v = gen()
        M = hebbian_write(M, k, v, lam)
        pairs.append({"k": k.tolist(), "v": v.tolist()})
        states.append(M.tolist())
    retrieval = read(M, np.array(pairs[2]["k"])).tolist()
    payload = {
        "params": {"d": d, "lam": lam, "seed": seed, "sparsity": sparsity,
                   "overlap": overlap, "nPairs": 6, "probeIndex": 2},
        "pairs": pairs,
        "states": states,
        "retrievalAtProbe": retrieval,
        "capacityHead": capacity_curve(d=24, lam=1.0, n_pairs=8),
        "decayHead": decay_curve(d=24, n_pairs=5, lam=0.9, seed=7),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=1)
    return payload


if __name__ == "__main__":
    print("CAPACITY CURVE (d=24, lam=1.0, sparsity=0.2, no overlap)")
    print("load | mean cosine | exact-match | interference share")
    for r in capacity_curve(n_pairs=30)[::3]:
        print(f"{r['load']:4d} |   {r['meanSim']:.3f}    |    {r['exactRate']:.3f}    |   {r['interferenceShare']:.3f}")
    print("\nDECAY CURVE (lam=0.9): similarity of the FIRST stored pair as more arrive")
    for r in decay_curve(n_pairs=6):
        print(f"  later pairs: {r['laterPairs']}   sim of first: {r['simFirst']:.3f}")
    export_vectors("expected_vectors.json")
    print("\nexported -> expected_vectors.json")
