"""Random map layouts for open_cleanup (2026-09-17).

Each layout has exactly the same number of cells of every kind as the mini map (12 x 16) and about the same
clustering, but the pieces sit in different places:

  river     32 cells: 26 start dirty ('F') + 6 start clean ('H'); a meandering 2-wide persistent random walk, one piece
  stream     5 cells ('~', always water), a single-width walk leaving the river
  orchard   48 cells ('B'), Eden growth biased along one axis, one piece, at least 7 steps from the river
  stations   8 cells, a chain with one empty cell between consecutive stations, no two touching (not even
             diagonally), between the river and the orchard; stored in chain order, so tools cycle T0..T3 along it
  spawns     7 cells on open ground, 1-7 steps from both the river and the orchard

A generated layout is kept only if its clustering and distance metrics are within `TOL` of the mini map's.
Layouts are generated once with a fixed seed and cached next to this file, so every run sees the same pool; the
pool is split 80 / 20 into training and held-out maps like the rule benchmark.

    python -m opensocialjax.environments.open_cleanup.layouts          # (re)build the cached pool, print stats
"""
from __future__ import annotations

import collections
import os
from dataclasses import dataclass

import numpy as np

H, W = 12, 16
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
KEYS = ("dirt", "potential_dirt", "river", "apple", "spawns", "tools")      # 'river' = the stream cells ('~' / 'S')
COUNTS = {"dirt": 26, "potential_dirt": 6, "river": 5, "apple": 48, "spawns": 7, "tools": 8}
POOL_SIZE, POOL_SEED, VERSION = 256, 0, 1
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"layouts_mini_v{VERSION}.npz")

MINI_MAP = [
    "FFFHFFFFHFFFFHFF",
    "FFHFFFFHFFFFHFFF",
    "  P   P    P ~  ",
    "             ~  ",
    "   P       P ~  ",
    "             ~  ",
    "             ~  ",
    "                ",
    "    P      P    ",
    "BBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBB",
    "BBBBBBBBBBBBBBBB",
]


# ---------------------------------------------------------------- metrics
def _cells(g, chars):
    return [tuple(int(v) for v in x) for x in np.argwhere(np.isin(g, list(chars)))]


def _dist(p, q):
    return abs(p[0] - q[0]) + abs(p[1] - q[1])


def _nearest(p, cs):
    return min(_dist(p, q) for q in cs)


def _adjacency(g, chars):
    cs = _cells(g, chars); s = set(cs); fr = []
    for r, c in cs:
        nb = [(r + dr, c + dc) for dr, dc in N4 if 0 <= r + dr < H and 0 <= c + dc < W]
        fr.append(sum(n in s for n in nb) / len(nb))
    return float(np.mean(fr))


def _components(g, chars):
    s = set(_cells(g, chars)); seen = set(); n = 0
    for x in s:
        if x in seen:
            continue
        n += 1; st = [x]; seen.add(x)
        while st:
            r, c = st.pop()
            for dr, dc in N4:
                y = (r + dr, c + dc)
                if y in s and y not in seen:
                    seen.add(y); st.append(y)
    return n


def metrics(g):
    """Counts, clustering and distances of a character grid ('F','H','~','B','T','P')."""
    river = _cells(g, "FH"); orch = _cells(g, "B"); st = _cells(g, "T"); sp = _cells(g, "P")
    return {"river_adj": _adjacency(g, "FH~"), "orch_adj": _adjacency(g, "B"),
            "river_comp": _components(g, "FH~"), "orch_comp": _components(g, "B"),
            "river_orch_min": min(_nearest(o, river) for o in orch),
            "orch_to_river": float(np.mean([_nearest(o, river) for o in orch])),
            "station_nn": float(np.mean([_nearest(p, [q for q in st if q != p]) for p in st])),
            "station_touch": any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) <= 1 for p in st for q in st if p != q),
            "station_to_river": float(np.mean([_nearest(p, river) for p in st])),
            "station_to_orch": float(np.mean([_nearest(p, orch) for p in st])),
            "spawn_to_river": float(np.mean([_nearest(p, river) for p in sp])),
            "spawn_to_orch": float(np.mean([_nearest(p, orch) for p in sp])),
            "counts": (len(_cells(g, "F")), len(_cells(g, "H")), len(_cells(g, "~")), len(orch), len(st), len(sp))}


def mini_grid():
    """The mini map as a character grid, stations included (row 6, even columns)."""
    g = np.array([list(r) for r in MINI_MAP])
    for c in range(0, W, 2):
        if g[6, c] == " ":
            g[6, c] = "T"
    return g


REFERENCE = metrics(mini_grid())
TOL = {"river_adj": 0.08, "orch_adj": 0.08, "river_orch_min": 1, "orch_to_river": 1.5, "station_to_river": 1.0,
       "station_to_orch": 1.0, "spawn_to_river": 1.5, "spawn_to_orch": 1.5}


def rejections(m):
    """Names of the checks a layout fails (empty = accepted)."""
    bad = [k for k, t in TOL.items() if abs(m[k] - REFERENCE[k]) > t]
    if m["river_comp"] != 1 or m["orch_comp"] != 1:
        bad.append("components")
    if m["station_touch"] or abs(m["station_nn"] - 2) > 1e-9:
        bad.append("station spacing")
    if m["counts"] != REFERENCE["counts"]:
        bad.append("counts")
    return bad


# ---------------------------------------------------------------- generator
class _Generator:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)
        self.fail = collections.Counter()

    def river(self, g):
        rng = self.rng
        for _ in range(50):
            r, c = int(rng.integers(0, H - 1)), int(rng.integers(0, W - 1)); d = N4[rng.integers(4)]; out = set()
            for _ in range(400):
                out |= {(r + dr, c + dc) for dr in (0, 1) for dc in (0, 1)}
                if len(out) >= 32:
                    break
                if rng.random() < 0.25:
                    d = N4[rng.integers(4)]
                nr, nc = r + d[0], c + d[1]
                if not (0 <= nr <= H - 2 and 0 <= nc <= W - 2):
                    d = N4[rng.integers(4)]; continue
                r, c = nr, nc
            if len(out) >= 32:
                for y in sorted(out, key=lambda y: rng.random())[:32]:
                    g[y] = "F"
                if _components(g, "F") == 1:
                    for y in sorted(_cells(g, "F"), key=lambda y: rng.random())[:6]:
                        g[y] = "H"
                    return True
                g[g == "F"] = " "
        return False

    def stream(self, g):
        rng = self.rng; river = _cells(g, "FH")
        for _ in range(30):
            r, c = river[rng.integers(len(river))]; d = N4[rng.integers(4)]; path = []
            for _ in range(40):
                nr, nc = r + d[0], c + d[1]
                if not (0 <= nr < H and 0 <= nc < W) or g[nr, nc] in "FH" or (nr, nc) in path:
                    d = N4[rng.integers(4)]; continue
                r, c = nr, nc; path.append((r, c))
                if len(path) == 5:
                    break
                if rng.random() < 0.2:
                    d = N4[rng.integers(4)]
            if len(path) == 5:
                for y in path:
                    g[y] = "~"
                return True
        return False

    def orchard(self, g):
        rng = self.rng; river = _cells(g, "FH")
        free = [y for y in _cells(g, " ") if _nearest(y, river) >= 7]
        if len(free) < 48:
            return False
        seed = free[rng.integers(len(free))]; s = {seed}; horiz = rng.random() < 0.5; ok = set(free)
        while len(s) < 48:
            front = sorted({(r + dr, c + dc) for r, c in s for dr, dc in N4} & ok - s)
            if not front:
                return False
            w = np.array([1.0 + 3.0 * ((abs(y[0] - seed[0]) < 2) if horiz else (abs(y[1] - seed[1]) < 2)) for y in front])
            s.add(front[rng.choice(len(front), p=w / w.sum())])
        for y in s:
            g[y] = "B"
        return True

    def stations(self, g):
        rng = self.rng; river, orch = _cells(g, "FH"), _cells(g, "B")

        def good(y, loose=False):
            if not (0 <= y[0] < H and 0 <= y[1] < W) or g[y] != " ":
                return False
            dr, do = _nearest(y, river), _nearest(y, orch)
            return (3 <= dr <= 7 and 2 <= do <= 5) if loose else (4 <= dr <= 6 and 2 <= do <= 4)

        steps = ((0, 2), (0, -2), (2, 0), (-2, 0))
        starts = [y for y in _cells(g, " ") if good(y)]
        for _ in range(60):
            if not starts:
                return None
            chain = [starts[rng.integers(len(starts))]]; d = steps[rng.integers(4)]
            while len(chain) < 8:
                placed = False
                for dd in sorted([d] * 3 + list(steps), key=lambda x: rng.random()):
                    y = (chain[-1][0] + dd[0], chain[-1][1] + dd[1])
                    mid = (chain[-1][0] + dd[0] // 2, chain[-1][1] + dd[1] // 2)
                    if good(y, loose=True) and g[mid] == " " and all(max(abs(y[0] - q[0]), abs(y[1] - q[1])) >= 2 for q in chain):
                        chain.append(y); d = dd; placed = True; break
                if not placed:
                    break
            if len(chain) == 8:
                for y in chain:
                    g[y] = "T"
                return chain
        return None

    def one(self):
        g = np.full((H, W), " ")
        for name, step in (("river", self.river), ("stream", self.stream), ("no room for the orchard", self.orchard)):
            if not step(g):
                self.fail[name] += 1
                return None, None
        chain = self.stations(g)
        if chain is None:
            self.fail["stations"] += 1
            return None, None
        river, orch = _cells(g, "FH"), _cells(g, "B")
        spots = [y for y in _cells(g, " ") if 1 <= _nearest(y, river) <= 7 and 1 <= _nearest(y, orch) <= 7]
        if len(spots) < 7:
            self.fail["spawns"] += 1
            return None, None
        for i in self.rng.choice(len(spots), 7, replace=False):
            g[spots[i]] = "P"
        return g, chain


def generate(n, seed=POOL_SEED, max_tries=100_000):
    """n accepted layouts as (char grid, station chain, metrics); plus generation / filter statistics."""
    gen = _Generator(seed); out = []; rejected = collections.Counter(); tries = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        g, chain = gen.one()
        if g is None:
            continue
        m = metrics(g); bad = rejections(m)
        rejected.update(bad)
        if not bad:
            out.append((g, chain, m))
    stats = {"tries": tries, "accepted": len(out), "generation_failures": dict(gen.fail), "filter_rejections": dict(rejected)}
    return out, stats


# ---------------------------------------------------------------- pool
@dataclass
class LayoutPool:
    """Stacked cell coordinates, one row per layout: arrays[k] has shape (n_layouts, COUNTS[k], 2)."""
    arrays: dict
    grid_shape: tuple = (H, W)

    @property
    def size(self):
        return int(self.arrays["dirt"].shape[0])

    def subset(self, idx):
        return LayoutPool({k: v[np.asarray(idx)] for k, v in self.arrays.items()}, self.grid_shape)

    def split(self, frac=0.8, seed=0):
        """(train, held_out) by a fixed shuffle, like RuleBenchmark.split."""
        perm = np.random.default_rng(seed).permutation(self.size); cut = int(round(frac * self.size))
        return self.subset(np.sort(perm[:cut])), self.subset(np.sort(perm[cut:]))

    def char_grid(self, i):
        g = np.full(self.grid_shape, " ")
        for key, ch in (("dirt", "F"), ("potential_dirt", "H"), ("river", "~"), ("apple", "B"), ("spawns", "P"), ("tools", "T")):
            for r, c in self.arrays[key][i]:
                g[r, c] = ch
        return g

    @staticmethod
    def from_generated(layouts):
        arrays = {k: [] for k in KEYS}
        for g, chain, _ in layouts:
            arrays["dirt"].append(_cells(g, "F")); arrays["potential_dirt"].append(_cells(g, "H"))
            arrays["river"].append(_cells(g, "~")); arrays["apple"].append(_cells(g, "B"))
            arrays["spawns"].append(_cells(g, "P")); arrays["tools"].append(list(chain))
        return LayoutPool({k: np.asarray(v, dtype=np.int16) for k, v in arrays.items()})

    def save(self, path):
        np.savez_compressed(path, version=VERSION, **self.arrays)

    @staticmethod
    def load(path):
        z = np.load(path)
        return LayoutPool({k: z[k] for k in KEYS})


def default_pool(rebuild=False):
    """The cached pool of POOL_SIZE layouts (generated with POOL_SEED on first use)."""
    if rebuild or not os.path.exists(CACHE):
        layouts, stats = generate(POOL_SIZE, POOL_SEED)
        assert len(layouts) == POOL_SIZE, stats
        LayoutPool.from_generated(layouts).save(CACHE)
        print(f"[layouts] wrote {CACHE}: {stats}")
    pool = LayoutPool.load(CACHE)
    for k, n in COUNTS.items():
        assert pool.arrays[k].shape[1:] == (n, 2), (k, pool.arrays[k].shape)
    return pool


def layout_split(on_train=False, frac=0.8, seed=0):
    train, held = default_pool().split(frac, seed)
    return train if on_train else held


if __name__ == "__main__":
    pool = default_pool(rebuild=True)
    ms = [metrics(pool.char_grid(i)) for i in range(pool.size)]
    print("reference:", {k: (round(v, 2) if isinstance(v, float) else v) for k, v in REFERENCE.items()})
    for k in TOL:
        v = np.array([m[k] for m in ms], dtype=float)
        print(f"  {k:17s} pool mean {v.mean():6.2f}  range {v.min():.2f}-{v.max():.2f}  (reference {REFERENCE[k]:.2f})")
    train, held = pool.split()
    print(f"pool {pool.size} layouts -> {train.size} train / {held.size} held out")
