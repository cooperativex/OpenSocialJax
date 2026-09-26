"""Random map layouts for open_harvest (2026-09-17).

The three patch SHAPES are kept exactly as on the mini map — one full 1-3-5-3-1 diamond of 13 trees and two
quarter diamonds of 6 — because regrowth reads the 12-cell neighbourhood of an empty cell, so a reshaped patch
would silently change the regrowth rates. What is drawn per layout is where each patch sits, how it is turned
(4 rotations x mirror), which slot gets the big one, and where the row of 13 spawn cells runs.

Invariants, all exact rather than within a tolerance:
  25 trees in 3 patches of 13 / 6 / 6, 13 spawn cells
  the neighbour-count profile of the trees: 4 cells with 3 neighbours, 8 with 4, 4 with 5, 8 with 7, 1 with 12
  patches at least 3 apart (Chebyshev), so no cell ever counts a neighbour from another patch
  spawn cells at least 2 away from every tree

    python -m opensocialjax.environments.open_harvest.layouts       # (re)build the cached pool, print stats
"""
from __future__ import annotations

import collections
import os
from dataclasses import dataclass

import numpy as np

H, W = 9, 15
N4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
NB12 = ((-1, 0), (1, 0), (0, -1), (0, 1), (-2, 0), (2, 0), (0, -2), (0, 2), (-1, -1), (-1, 1), (1, -1), (1, 1))
SLOT_LETTERS = "RYG"                     # patch slots as the env's map parser reads them (the colours are drawn per episode)
N_TREES, N_SPAWN, N_PATCH = 25, 13, 3
POOL_SIZE, POOL_SEED, VERSION = 256, 0, 1
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"layouts_mini_v{VERSION}.npz")

MINI_MAP = [
    "RRR         GGG",
    "RR     Y     GG",
    "R     YYY     G",
    "     YYYYY     ",
    "      YYY      ",
    "       Y       ",
    "               ",
    "  QQQQQQQQQQQ  ",
    " PPPPPPPPPPPPP ",
]
DIAMOND = ((0, 2), (1, 1), (1, 2), (1, 3), (2, 0), (2, 1), (2, 2), (2, 3), (2, 4), (3, 1), (3, 2), (3, 3), (4, 2))
QUARTER = ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (2, 0))


def _orient(cells, k, mirror):
    """cells turned k quarter turns and optionally mirrored, normalised to start at (0, 0)."""
    out = []
    for r, c in cells:
        for _ in range(k):
            r, c = c, -r
        out.append((r, -c) if mirror else (r, c))
    r0 = min(r for r, _ in out); c0 = min(c for _, c in out)
    return tuple(sorted((r - r0, c - c0) for r, c in out))


ORIENTATIONS = {"diamond": sorted({_orient(DIAMOND, k, m) for k in range(4) for m in (False, True)}),
                "quarter": sorted({_orient(QUARTER, k, m) for k in range(4) for m in (False, True)})}


def neighbour_profile(trees):
    s = set(trees)
    return dict(sorted(collections.Counter(sum(1 for dr, dc in NB12 if (r + dr, c + dc) in s) for r, c in s).items()))


def mini_grid():
    return np.array([list(r) for r in MINI_MAP])


def metrics(g):
    trees = {(r, c): g[r, c] for r in range(H) for c in range(W) if g[r, c] in SLOT_LETTERS}
    by = collections.defaultdict(list)
    for rc, ch in trees.items():
        by[ch].append(rc)
    seps = [min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p in by[a] for q in by[b])
            for i, a in enumerate(SLOT_LETTERS) for b in SLOT_LETTERS[i + 1:] if a in by and b in by]
    spawns = [(r, c) for r in range(H) for c in range(W) if g[r, c] == "P"]
    return {"trees": len(trees), "patch_sizes": sorted(len(v) for v in by.values()), "spawns": len(spawns),
            "neighbours": neighbour_profile(trees), "min_patch_separation": min(seps) if seps else None,
            "spawn_to_tree": min(max(abs(p[0] - q[0]), abs(p[1] - q[1])) for p in spawns for q in trees) if spawns and trees else None}


REFERENCE = metrics(mini_grid())


def _place(rng, taken, shape_name):
    """A random orientation and position of a shape that keeps >= 3 cells (Chebyshev) from everything placed."""
    shapes = ORIENTATIONS[shape_name]
    for _ in range(200):
        cells = shapes[rng.integers(len(shapes))]
        hh = max(r for r, _ in cells) + 1; ww = max(c for _, c in cells) + 1
        if hh > H or ww > W:
            continue
        r0 = int(rng.integers(0, H - hh + 1)); c0 = int(rng.integers(0, W - ww + 1))
        put = [(r + r0, c + c0) for r, c in cells]
        if any(max(abs(p[0] - q[0]), abs(p[1] - q[1])) < 3 for p in put for q in taken):
            continue
        return put
    return None


def generate_one(rng):
    """(char grid, patch cells per slot) or (None, None)."""
    big = int(rng.integers(N_PATCH))                       # which slot gets the 13-tree diamond
    taken = []; patches = [None] * N_PATCH
    for slot in [big] + [s for s in range(N_PATCH) if s != big]:   # the diamond first: placed last it rarely fits, which
        put = _place(rng, taken, "diamond" if slot == big else "quarter")   # would bias which slot ends up big
        if put is None:
            return None, None
        taken += put; patches[slot] = put
    g = np.full((H, W), " ")
    for slot, cells in enumerate(patches):
        for rc in cells:
            g[rc] = SLOT_LETTERS[slot]
    # spawn cells: a straight run of 13, at least 2 away from every tree
    runs = []
    for r in range(H):
        for c0 in range(W - N_SPAWN + 1):
            run = [(r, c) for c in range(c0, c0 + N_SPAWN)]
            if all(g[rc] == " " for rc in run) and all(max(abs(p[0] - q[0]), abs(p[1] - q[1])) >= 2 for p in run for q in taken):
                runs.append(run)
    if not runs:
        return None, None
    for rc in runs[rng.integers(len(runs))]:
        g[rc] = "P"
    return g, patches


def generate(n, seed=POOL_SEED, max_tries=100_000):
    rng = np.random.default_rng(seed); out = []; tries = 0; fails = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        g, patches = generate_one(rng)
        if g is None:
            fails += 1
            continue
        m = metrics(g)
        assert m["neighbours"] == REFERENCE["neighbours"], (m["neighbours"], REFERENCE["neighbours"])
        assert m["trees"] == N_TREES and m["spawns"] == N_SPAWN and sorted(m["patch_sizes"]) == sorted(REFERENCE["patch_sizes"])
        out.append((g, patches, m))
    return out, {"tries": tries, "accepted": len(out), "placement_failures": fails}


@dataclass
class LayoutPool:
    """apple: (n, 25, 2) tree cells; slot: (n, 25) which patch each tree belongs to; spawns: (n, 13, 2)."""
    arrays: dict
    grid_shape: tuple = (H, W)

    @property
    def size(self):
        return int(self.arrays["apple"].shape[0])

    def subset(self, idx):
        return LayoutPool({k: v[np.asarray(idx)] for k, v in self.arrays.items()}, self.grid_shape)

    def split(self, frac=0.8, seed=0):
        perm = np.random.default_rng(seed).permutation(self.size); cut = int(round(frac * self.size))
        return self.subset(np.sort(perm[:cut])), self.subset(np.sort(perm[cut:]))

    def char_grid(self, i):
        g = np.full(self.grid_shape, " ")
        for (r, c), s in zip(self.arrays["apple"][i], self.arrays["slot"][i]):
            g[r, c] = SLOT_LETTERS[int(s)]
        for r, c in self.arrays["spawns"][i]:
            g[r, c] = "P"
        return g

    def map_ascii(self, i):
        return ["".join(row) for row in self.char_grid(i)]

    @staticmethod
    def from_generated(layouts):
        apple, slot, spawns = [], [], []
        for g, patches, _ in layouts:
            cells = []; slots = []
            for s, ps in enumerate(patches):
                cells += list(ps); slots += [s] * len(ps)
            order = sorted(range(len(cells)), key=lambda k: cells[k])          # row-major, like find_positions
            apple.append([cells[k] for k in order]); slot.append([slots[k] for k in order])
            spawns.append([(int(r), int(c)) for r in range(H) for c in range(W) if g[r, c] == "P"])
        return LayoutPool({"apple": np.asarray(apple, dtype=np.int16), "slot": np.asarray(slot, dtype=np.int16),
                           "spawns": np.asarray(spawns, dtype=np.int16)})

    def save(self, path):
        np.savez_compressed(path, version=VERSION, **self.arrays)

    @staticmethod
    def load(path):
        z = np.load(path)
        return LayoutPool({k: z[k] for k in ("apple", "slot", "spawns")})


def default_pool(rebuild=False):
    if rebuild or not os.path.exists(CACHE):
        layouts, stats = generate(POOL_SIZE, POOL_SEED)
        assert len(layouts) == POOL_SIZE, stats
        LayoutPool.from_generated(layouts).save(CACHE)
        print(f"[layouts] wrote {CACHE}: {stats}")
    pool = LayoutPool.load(CACHE)
    assert pool.arrays["apple"].shape[1:] == (N_TREES, 2) and pool.arrays["spawns"].shape[1:] == (N_SPAWN, 2)
    return pool


def layout_split(on_train=False, frac=0.8, seed=0):
    train, held = default_pool().split(frac, seed)
    return train if on_train else held


if __name__ == "__main__":
    pool = default_pool(rebuild=True)
    ms = [metrics(pool.char_grid(i)) for i in range(pool.size)]
    print("reference:", REFERENCE)
    assert all(m["neighbours"] == REFERENCE["neighbours"] for m in ms), "a layout changed the regrowth neighbourhoods"
    seps = [m["min_patch_separation"] for m in ms]; s2t = [m["spawn_to_tree"] for m in ms]
    print(f"pool {pool.size} layouts | patch separation {min(seps)}-{max(seps)} (mini map {REFERENCE['min_patch_separation']})"
          f" | spawn-to-tree {min(s2t)}-{max(s2t)} (mini map {REFERENCE['spawn_to_tree']})")
    big = collections.Counter(int(np.bincount(pool.arrays["slot"][i]).argmax()) for i in range(pool.size))
    print("slot holding the 13-tree diamond:", dict(sorted(big.items())))
    train, held = pool.split()
    print(f"-> {train.size} train / {held.size} held out")
    print("\n".join("".join(r) for r in pool.char_grid(0)))
