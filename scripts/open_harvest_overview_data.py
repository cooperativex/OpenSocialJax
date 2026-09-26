"""Facts and the example map for the Harvest overview figure (make_harvest_overview_figure.py), for one seed of the
formal experiment: the seed's map (layouts_orig6 held-out pool, pick_layout), the colour each of its six patches drew,
its hidden rule set (pay and regrowth speed per colour and stage), and a mid-game picture of that map -- the real
layout and colours, with an illustrative mix of ripeness stages, some trees eaten and five agents placed on the ground.
Needs JAX (on the login node: see the login-node JAX notes).  Writes docs/figures/harvest_overview_seed<k>.png and
scripts/open_harvest_overview_data.json.

usage: python scripts/open_harvest_overview_data.py [SEED]"""
import json, os, sys
sys.path.insert(0, ".")
import jax, jax.numpy as jnp, numpy as onp
from PIL import Image
from llm_policy import open_harvest_policy as H
from opensocialjax.environments.open_harvest.layouts_orig6 import pick_layout, layout_split
from opensocialjax.environments.open_harvest.open_harvest import Items as HItems, APPLE_BASE, NUM_RIPE_STAGES, HUE_NAMES
from opensocialjax.environments.discovery_rules import STAGE_PAY, REGROW_RATES

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 906
rng = onp.random.default_rng(7)
layout = pick_layout(SEED)
env = H.make_env(5, 300, 5, 25, False, layout, 12)
st = H.initial_state(env, SEED, 0)
hues = onp.array(st.cell_hues); trees = [tuple(int(x) for x in rc) for rc in onp.array(env.SPAWNS_APPLE)]
# hidden rules of this seed, as the env holds them
pay = [[float(x) for x in row] for row in onp.array(env.pay_table(st))] if hasattr(env, "pay_table") else None
truth = H.truth_of(env, st) if hasattr(H, "truth_of") else None

# the patches: trees grouped by connected component (patches are >= 3 apart), with the colour each drew
left, patches = set(trees), []
while left:
    stack = [left.pop()]; comp = [stack[0]]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                n = (r + dr, c + dc)
                if n in left:
                    left.remove(n); stack.append(n); comp.append(n)
    k = trees.index(comp[0]); patches.append({"hue": HUE_NAMES[int(hues[k])], "trees": len(comp), "top_left": [min(p[0] for p in comp), min(p[1] for p in comp)]})
patches.sort(key=lambda p: (p["top_left"][0], p["top_left"][1]))

# a mid-game picture: about a third of the trees eaten, a mix of stages, agents on the ground
g = onp.array(st.grid); floor = int(HItems.empty); stage_p = [0.35, 0.40, 0.25]
for k, rc in enumerate(trees):
    g[rc] = floor if rng.random() < 0.3 else APPLE_BASE + int(hues[k]) * NUM_RIPE_STAGES + int(rng.choice(3, p=stage_p))
locs = onp.array(st.agent_locs)
for i in range(len(locs)):
    g[locs[i][0], locs[i][1]] = floor
opts = [(r, c) for r in range(1, g.shape[0] - 1) for c in range(1, g.shape[1] - 1) if g[r, c] == floor and (r, c) not in set(trees)
        and min(abs(r - a) + abs(c - b) for a, b in trees) <= 2]
pick = [opts[int(i)] for i in rng.choice(len(opts), size=len(locs), replace=False)]
for i, (r, c) in enumerate(pick):
    g[r, c] = len(HItems) + i; locs[i] = (r, c, int(rng.integers(4)))
st = st.replace(grid=jnp.array(g, dtype=st.grid.dtype), agent_locs=jnp.array(locs, dtype=st.agent_locs.dtype))
img = Image.fromarray(onp.asarray(env.render(st)).astype(onp.uint8))
out_png = f"docs/figures/harvest_overview_seed{SEED}.png"
img.resize((img.width * 4, img.height * 4), Image.NEAREST).save(out_png)

# the hidden rule set, read from a finished formal run of this seed (every model met the same one)
r = json.load(open(f"experiments/open_harvest_5ag/runs/gpt-6-luna/seed{SEED}/result.json"))
assert r["layout"] == layout, (r["layout"], layout)
data = {"seed": SEED, "layout": layout, "pool_size": int(layout_split(False).size + layout_split(True).size),
        "held_out": int(layout_split(False).size), "hues": list(HUE_NAMES), "patches": patches,
        "pay": r["truth"]["pay"], "speed": r["truth"]["speed"], "stage_pay": list(STAGE_PAY), "rates": list(REGROW_RATES),
        "map_png": out_png, "grid": [int(x) for x in g.shape]}
json.dump(data, open("scripts/open_harvest_overview_data.json", "w"), indent=1)
print(json.dumps({k: v for k, v in data.items() if k != "patches"}), "\npatches:", patches)
