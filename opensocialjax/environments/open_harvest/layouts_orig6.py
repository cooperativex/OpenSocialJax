"""open_harvest on the original Commons Harvest map: 16 x 22, six patches (2026-09-24).

ORIG6_MAP is the original Commons Harvest map (opensocialjax/environments/common_harvest/harvest_open.py) as it
is: four 1-3-5-3-1 diamonds and two quarter diamonds, 64 trees (12.8 each for 5 agents), the original spawn rows.
Each patch is its own map letter (slots R Y G C M S), and the env is built with patch_hues_with_replacement, so a
meta-episode draws each patch's hue independently from the five: two patches can share a colour and a colour can be
missing. The hues are kept across the episode's trials.

The random pool keeps the patch SHAPES for the reason given in layouts.py (regrowth reads the 12-cell neighbourhood
of an empty cell, so reshaped patches would silently change the regrowth rates); what is drawn per layout is where
each patch sits and how it is turned, and where a straight run of spawn cells lies.

Invariants, all exact:
  64 trees in 6 patches of 13 / 13 / 13 / 13 / 6 / 6, one slot per patch
  the neighbour-count profile of the trees equals ORIG6_MAP's
  patches at least 3 apart (Chebyshev), so no cell ever counts a neighbour from another patch
  N_SPAWN spawn cells in a straight run, at least MIN_SPAWN_TO_TREE (4) away from every tree -- the original map's
  own distance (its bottom spawn row is 4 from the two middle diamonds)

    python -m opensocialjax.environments.open_harvest.layouts_orig6    # (re)build the cached pool, print stats
"""
from __future__ import annotations

import os

import numpy as np

from opensocialjax.environments.open_harvest.layouts import ORIENTATIONS, LayoutPool, neighbour_profile

H, W = 16, 22
SHAPES = ("diamond", "diamond", "diamond", "diamond", "quarter", "quarter")
SLOT_LETTERS = "RYGCMS"                  # one slot per patch, as the env's map parser reads them (index into RYGCMS)
N_TREES, N_SPAWN, MIN_SPAWN_TO_TREE = 64, 20, 4
POOL_SIZE, POOL_SEED, VERSION = 256, 0, 1
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"layouts_orig6_v{VERSION}.npz")

ORIG6_MAP = [
    "RRR    Y      G    CCC",
    "RR    YYY    GGG    CC",
    "R    YYYYY  GGGGG    C",
    "      YYY    GGG      ",
    "       Y      G       ",
    "  M                S  ",
    " MMM  Q        Q  SSS ",
    "MMMMM            SSSSS",
    " MMM              SSS ",
    "  M                S  ",
    "                      ",
    "                      ",
    "                      ",
    "  PPPPPPPPPPPPPPPPPP  ",
    " PPPPPPPPPPPPPPPPPPPP ",
    "PPPPPPPPPPPPPPPPPPPPPP",
]


def _cheb(p, q):
    return max(abs(p[0] - q[0]), abs(p[1] - q[1]))


def patches_of(trees, gap=2):
    """Group tree cells into patches: cells within `gap` (Chebyshev) of each other are one patch. Patches are >= 3
    apart, so with the default gap two patches never merge, whatever their colours."""
    left, out = set(trees), []
    while left:
        stack = [left.pop()]; group = []
        while stack:
            p = stack.pop(); group.append(p)
            for q in [q for q in left if _cheb(p, q) <= gap]:
                left.discard(q); stack.append(q)
        out.append(sorted(group))
    return sorted(out)


def metrics(g):
    trees = [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] in SLOT_LETTERS]
    ps = patches_of(trees, gap=1)
    spawns = [(r, c) for r in range(g.shape[0]) for c in range(g.shape[1]) if g[r, c] == "P"]
    seps = [min(_cheb(p, q) for p in a for q in b) for i, a in enumerate(ps) for b in ps[i + 1:]]
    return {"trees": len(trees), "patch_sizes": sorted(len(p) for p in ps), "spawns": len(spawns),
            "neighbours": neighbour_profile(trees), "min_patch_separation": min(seps) if seps else None,
            "spawn_to_tree": min(_cheb(p, q) for p in spawns for q in trees) if spawns and trees else None}


REFERENCE = metrics(np.array([list(r) for r in ORIG6_MAP]))


def _place(rng, taken, shape_name):
    """A random orientation and position of a shape that keeps >= 3 cells (Chebyshev) from everything placed."""
    shapes = ORIENTATIONS[shape_name]
    for _ in range(200):
        cells = shapes[rng.integers(len(shapes))]
        hh = max(r for r, _ in cells) + 1; ww = max(c for _, c in cells) + 1
        r0 = int(rng.integers(0, H - hh + 1)); c0 = int(rng.integers(0, W - ww + 1))
        put = [(r + r0, c + c0) for r, c in cells]
        if any(_cheb(p, q) < 3 for p in put for q in taken):
            continue
        return put
    return None


def generate_one(rng):
    """A char grid, or None."""
    taken = []; g = np.full((H, W), " ")
    for slot, shape in enumerate(SHAPES):                  # the diamonds first: placed last they would rarely fit
        put = _place(rng, taken, shape)
        if put is None:
            return None
        taken += put
        for rc in put:
            g[rc] = SLOT_LETTERS[slot]
    runs = []                                              # spawn cells: a straight run, at least MIN_SPAWN_TO_TREE from every tree
    for r in range(H):
        for c0 in range(W - N_SPAWN + 1):
            run = [(r, c) for c in range(c0, c0 + N_SPAWN)]
            if all(g[rc] == " " for rc in run) and all(_cheb(p, q) >= MIN_SPAWN_TO_TREE for p in run for q in taken):
                runs.append(run)
    if not runs:
        return None
    for rc in runs[rng.integers(len(runs))]:
        g[rc] = "P"
    return g


def generate(n, seed=POOL_SEED, max_tries=2_000_000):
    rng = np.random.default_rng(seed); out = []; tries = fails = 0
    while len(out) < n and tries < max_tries:
        tries += 1
        g = generate_one(rng)
        if g is None:
            fails += 1
            continue
        m = metrics(g)
        assert m["neighbours"] == REFERENCE["neighbours"], (m["neighbours"], REFERENCE["neighbours"])
        assert m["trees"] == N_TREES and m["spawns"] == N_SPAWN and m["patch_sizes"] == REFERENCE["patch_sizes"]
        out.append(g)
    return out, {"tries": tries, "accepted": len(out), "placement_failures": fails}


class Orig6Pool(LayoutPool):
    """LayoutPool with six slot letters (the base class draws three)."""

    def subset(self, idx):
        return Orig6Pool({k: v[np.asarray(idx)] for k, v in self.arrays.items()}, self.grid_shape)

    def char_grid(self, i):
        g = np.full(self.grid_shape, " ")
        for (r, c), s in zip(self.arrays["apple"][i], self.arrays["slot"][i]):
            g[r, c] = SLOT_LETTERS[int(s)]
        for r, c in self.arrays["spawns"][i]:
            g[r, c] = "P"
        return g


def _to_pool(grids):
    cells = [[(r, c) for r in range(H) for c in range(W) if g[r, c] in SLOT_LETTERS] for g in grids]   # row-major, like find_positions
    slots = [[SLOT_LETTERS.index(g[rc]) for rc in cs] for g, cs in zip(grids, cells)]
    spawns = [[(r, c) for r in range(H) for c in range(W) if g[r, c] == "P"] for g in grids]
    return Orig6Pool({"apple": np.asarray(cells, dtype=np.int16), "slot": np.asarray(slots, dtype=np.int16),
                      "spawns": np.asarray(spawns, dtype=np.int16)}, (H, W))


def default_pool(rebuild=False):
    if rebuild or not os.path.exists(CACHE):
        grids, stats = generate(POOL_SIZE, POOL_SEED)
        assert len(grids) == POOL_SIZE, stats
        _to_pool(grids).save(CACHE)
        print(f"[layouts_orig6] wrote {CACHE}: {stats}")
    pool = Orig6Pool(LayoutPool.load(CACHE).arrays, (H, W))
    assert pool.arrays["apple"].shape[1:] == (N_TREES, 2) and pool.arrays["spawns"].shape[1:] == (N_SPAWN, 2)
    return pool


def layout_split(on_train=False, frac=0.8, seed=0):
    train, held = default_pool().split(frac, seed)
    return train if on_train else held


def pick_layout(seed, on_train=False):
    """The map a meta-episode with this seed plays on: an index into the held-out (or training) split, fixed by the seed,
    so every model evaluated on a seed meets the same map."""
    return int(np.random.default_rng(seed).integers(layout_split(on_train).size))


if __name__ == "__main__":
    pool = default_pool(rebuild=True)
    ms = [metrics(pool.char_grid(i)) for i in range(pool.size)]
    print("reference (ORIG6_MAP):", REFERENCE)
    assert all(m["neighbours"] == REFERENCE["neighbours"] for m in ms), "a layout changed the regrowth neighbourhoods"
    seps = [m["min_patch_separation"] for m in ms]; s2t = [m["spawn_to_tree"] for m in ms]
    print(f"pool {pool.size} layouts | patch separation {min(seps)}-{max(seps)} (ORIG6_MAP {REFERENCE['min_patch_separation']})"
          f" | spawn-to-tree {min(s2t)}-{max(s2t)} (ORIG6_MAP {REFERENCE['spawn_to_tree']})")
    train, held = pool.split()
    print(f"-> {train.size} train / {held.size} held out")
    print("\n".join("".join(r) for r in pool.char_grid(0)))
