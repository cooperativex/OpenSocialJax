"""Mechanism checks for the open_cleanup hidden-rule environment.

The XLand-style twist is that the river carries several visually distinct
waste types, three tools lie on the ground, and a rule drawn per meta-episode
— never shown to the agents — decides which tool clears which waste type.
Firing with the wrong tool does nothing at all, so the rule cannot be ignored.

These checks pin that down: the right tool must clear and the wrong tool must
not, tools must be pickable and non-consumable, the rule must vary across
meta-episodes while staying fixed across the trials inside one, and pollution
must stay exactly as in vanilla Clean Up (every waste type counts).

Run:
  ulimit -c 0; ulimit -u $(ulimit -Hu)
  export PYTHONPATH=$PWD:$PYTHONPATH
  JAX_PLATFORMS=cpu python tests/test_open_cleanup.py
"""

import sys

import jax
import jax.numpy as jnp
import numpy as onp

import opensocialjax
from opensocialjax.environments.open_cleanup.open_cleanup import (
    DIRT_TYPES, NUM_DIRT_TYPES, NUM_TOOLS, NUM_COLORS, RULE_LEN, Actions, Items,
    all_rules, all_rulesets, sample_rule, sample_ruleset, rule_table,
)
from opensocialjax.environments.discovery_rules import (
    EMPTY, MAX_RULES, RULE_ENCODING_LEN, TOOL_CLEARS, TOOL_TRANSFORMS,
    check_clean, decode_chain, decode_tool_table, encode_tool_table,
    chain_depth_of,
)
from opensocialjax.environments.open_cleanup.open_cleanup import (
    rule_next, sample_chain_ruleset, sample_bijective_ruleset,
)
from opensocialjax.environments.rule_benchmark import enumerate_benchmark

STEPS = 150


def place_agent(state, env, row, col, facing, tool=None):
    """Move agent 0 to (row, col) facing `facing`, optionally arming it.

    `_step` starts by overwriting agent_locs with reborn_locs, so both have
    to be set; the grid stamp is refreshed too so no stale agent lingers."""
    old = onp.asarray(state.agent_locs)[0]
    grid = state.grid.at[int(old[0]), int(old[1])].set(jnp.int16(Items.empty))
    grid = grid.at[row, col].set(jnp.int16(onp.asarray(env._agents)[0]))
    row_vec = jnp.array([row, col, facing], dtype=state.agent_locs.dtype)
    state = state.replace(
        grid=grid,
        agent_locs=state.agent_locs.at[0].set(row_vec),
        reborn_locs=state.reborn_locs.at[0].set(row_vec),
    )
    if tool is not None:
        state = state.replace(held_tool=state.held_tool.at[0].set(jnp.int16(tool)))
    return state


def try_clean(env, base, target, tool):
    """Stand agent 0 next to `target` holding `tool` and fire from every
    adjacent cell/facing. Returns True if the cell ever became clean water."""
    step = jax.jit(env.step)
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        r, c = int(target[0]) + dr, int(target[1]) + dc
        if not (0 <= r < env.GRID_SIZE_ROW and 0 <= c < env.GRID_SIZE_COL):
            continue
        for facing in range(4):
            st = place_agent(base, env, r, c, facing, tool)
            acts = [int(Actions.zap_clean)] + \
                   [int(Actions.stay)] * (env.num_agents - 1)
            _, st, _, _, _ = step(jax.random.PRNGKey(0), st, acts)
            # the waste arrays are re-sorted every step, so look the cell up
            # by coordinate (the grid is re-stamped from the labels)
            cell = int(onp.asarray(st.grid)[int(target[0]), int(target[1])])
            if cell == int(Items.potential_dirt):
                return True
    return False


def test_only_the_matching_tool_clears_a_waste_type():
    """The whole design rests on this: the right tool clears, every other tool
    does nothing whatsoever. A wrong tool that merely cleaned *slower* is what
    made the previous version's rule not worth inferring."""
    env = opensocialjax.make("open_cleanup")
    _, base = env.reset(jax.random.PRNGKey(3))
    rule = onp.asarray(rule_table(base))
    labels = onp.asarray(base.potential_dirt_and_dirt_label)
    locs = onp.asarray(base.potential_dirt_and_dirt_locs)

    for k in range(NUM_DIRT_TYPES):
        wtype = int(DIRT_TYPES[k])
        idx = onp.flatnonzero(labels == wtype)
        assert idx.size > 0, f"waste type {wtype} never spawned at reset"
        target = locs[idx[0]]

        right = int(rule[k])
        assert try_clean(env, base, target, right), (
            f"waste type {k} could not be cleared with its own tool {right} "
            f"from any adjacent cell/facing (rule={rule.tolist()})")
        for wrong in range(NUM_TOOLS):
            if wrong == right:
                continue
            assert not try_clean(env, base, target, wrong), (
                f"waste type {k} was cleared by tool {wrong}, but rule says "
                f"only tool {right} works — the rule is not gating cleaning")
    print(f"ok: rule={rule.tolist()} — each waste type yields to its own tool "
          f"and to no other")


def test_stepping_on_a_station_picks_the_tool_up_and_leaves_it_there():
    """Tools are a choice, not a resource: walking onto a station arms the
    agent and the station stays put for whoever comes next."""
    env = opensocialjax.make("open_cleanup")
    _, base = env.reset(jax.random.PRNGKey(5))
    step = jax.jit(env.step)
    stations = onp.asarray(env.TOOLS)
    station_ids = onp.asarray(env._tool_ids)

    for st_idx in range(len(stations)):
        row, col = int(stations[st_idx][0]), int(stations[st_idx][1])
        want = int(station_ids[st_idx])
        # start one cell away holding a different tool, then walk on
        start_row = row - 1
        other = (want + 1) % NUM_TOOLS
        st = place_agent(base, env, start_row, col, 0, other)
        assert int(onp.asarray(st.held_tool)[0]) == other

        # STEP_MOVE is world-frame: Actions.up increases the row
        acts = [int(Actions.up)] + [int(Actions.stay)] * (env.num_agents - 1)
        _, st, _, _, _ = step(jax.random.PRNGKey(11), st, acts)
        loc = onp.asarray(st.agent_locs)[0]
        if (int(loc[0]), int(loc[1])) != (row, col):
            continue          # blocked this step; other stations still test it
        assert int(onp.asarray(st.held_tool)[0]) == want, (
            f"stepping onto station {st_idx} (tool {want}) left the agent "
            f"holding {int(onp.asarray(st.held_tool)[0])}")

        # walk off again — the station must still be on the grid
        acts = [int(Actions.down)] + [int(Actions.stay)] * (env.num_agents - 1)
        _, st, _, _, _ = step(jax.random.PRNGKey(12), st, acts)
        assert int(onp.asarray(st.grid)[row, col]) == int(env._tool_labels[st_idx]), (
            f"station {st_idx} vanished after an agent walked over it — tool "
            f"cells must not be consumed")
        print(f"ok: station {st_idx} at ({row},{col}) gives tool {want} and stays")
        return
    raise AssertionError("no station was reachable in one step from above")


def test_the_held_tool_is_visible_in_the_observation():
    """An agent that cannot tell what it is holding cannot apply the rule at
    all, which would make the task unsolvable rather than hard."""
    env = opensocialjax.make("open_cleanup")
    _, base = env.reset(jax.random.PRNGKey(0))
    step = jax.jit(env.step)
    seen = []
    for tool in range(NUM_TOOLS):
        # obs is only produced by reset/step, so arm the agent and take one
        # turn-in-place step — that cannot walk it onto a station
        st = base.replace(
            held_tool=jnp.full_like(base.held_tool, jnp.int16(tool)))
        obs, st, _, _, _ = step(
            jax.random.PRNGKey(8), st,
            [int(Actions.turn_left)] * env.num_agents)
        arr = onp.asarray(obs[0])
        # the held tool is a one-hot over the shared COLOUR axis (XLand
        # style), the last NUM_COLORS channels
        seen.append(arr[..., -NUM_COLORS:].reshape(-1, NUM_COLORS)[0].copy())
    for tool in range(NUM_TOOLS):
        expect = onp.zeros(NUM_COLORS)
        expect[tool] = 1
        assert onp.array_equal(seen[tool], expect), (
            f"holding tool {tool} gave obs channels {seen[tool].tolist()}, "
            f"expected the one-hot {expect.tolist()}")
    print(f"ok: the held tool reads off the last {NUM_COLORS} obs channels")


def test_every_waste_type_pollutes():
    """The hidden rule decides which TOOL clears a type, not whether the type
    matters. Pollution must stay exactly as in vanilla Clean Up, otherwise the
    inherited thresholdDepletion is no longer calibrated."""
    env = opensocialjax.make("open_cleanup")
    step = jax.jit(env.step)

    def apples_after_clearing(types):
        _, state = env.reset(jax.random.PRNGKey(4))
        labels = state.potential_dirt_and_dirt_label
        locs = state.potential_dirt_and_dirt_locs
        hit = jnp.isin(labels, jnp.array(types, dtype=labels.dtype))
        new_labels = jnp.where(hit, jnp.int16(Items.potential_dirt), labels)
        state = state.replace(
            potential_dirt_and_dirt_label=new_labels,
            grid=state.grid.at[locs[:, 0], locs[:, 1]].set(new_labels))
        key, total = jax.random.PRNGKey(99), 0
        for _ in range(STEPS):
            key, ks = jax.random.split(key)
            _, state, _, _, _ = step(
                ks, state, [int(Actions.turn_left)] * env.num_agents)
            total += int(onp.sum(onp.asarray(state.grid) == int(Items.apple)))
        return total

    types = [int(t) for t in DIRT_TYPES]
    none = apples_after_clearing([])
    everything = apples_after_clearing(types)
    assert everything > 3 * max(none, 1), (
        f"clearing all waste gave {everything} apple-steps vs {none} for "
        f"clearing nothing — pollution is not gating apple growth at all")

    # every individual type must contribute: clearing five of six must leave
    # the river visibly dirtier than clearing all six
    for k in range(NUM_DIRT_TYPES):
        partial = apples_after_clearing([t for t in types if t != types[k]])
        assert partial < everything, (
            f"leaving waste type {k} behind cost nothing ({partial} vs "
            f"{everything} apple-steps) — that type does not pollute")
    print(f"ok: all {NUM_DIRT_TYPES} waste types pollute "
          f"(none={none}, all cleared={everything})")


def _space_rules():
    """The enumerated table space, or a 100k sample of it when enumeration
    would not fit a login node (3**15 tables)."""
    from opensocialjax.environments.open_cleanup.open_cleanup import (
        rule_space_size, ENUMERABLE_LIMIT)
    if rule_space_size() > ENUMERABLE_LIMIT:
        keys = jax.random.split(jax.random.PRNGKey(0), 100_000)
        return onp.asarray(jax.vmap(sample_rule)(keys)), False
    return onp.asarray(all_rules()), True


def test_rule_space_is_the_full_tool_assignment_space():
    env = opensocialjax.make("open_cleanup")
    rules, full = _space_rules()
    if full:
        assert rules.shape == (NUM_TOOLS ** NUM_DIRT_TYPES, RULE_LEN), rules.shape
        assert len(set(map(tuple, rules.tolist()))) == rules.shape[0], \
            "all_rules() contains duplicates — a train/test split would leak"
        drawn = {tuple(onp.asarray(sample_rule(jax.random.PRNGKey(s))).tolist())
                 for s in range(200)}
        assert drawn <= set(map(tuple, rules.tolist())), \
            "sample_rule() can draw rules that all_rules() does not enumerate"
    else:
        assert rules.shape[1] == RULE_LEN and ((rules >= 0) & (rules < NUM_TOOLS)).all()

    seen = set()
    for seed in range(30):
        _, state = env.reset(jax.random.PRNGKey(seed))
        enc = onp.asarray(state.rule_encoding)
        assert enc.shape == (MAX_RULES, RULE_ENCODING_LEN), enc.shape
        r = onp.asarray(rule_table(state))
        assert r.shape == (RULE_LEN,), r.shape
        assert ((r >= 0) & (r < NUM_TOOLS)).all(), f"tool out of range: {r}"
        seen.add(tuple(r.tolist()))
    assert len(seen) >= 20, f"rule space barely explored across resets: {len(seen)}"
    print(f"ok: {NUM_TOOLS ** NUM_DIRT_TYPES} rules in the space "
          f"({'enumerated' if full else 'sampled'}), {len(seen)} distinct across 30 resets")


def test_rule_is_shared_across_trials_of_a_meta_episode():
    """A meta-episode is several trials sharing one hidden rule: the layout
    restarts between trials but the rule must not, otherwise an agent can
    never use what it worked out in an earlier trial."""
    inner, outer = 30, 4
    env = opensocialjax.make("open_cleanup",
                             num_inner_steps=inner, num_outer_steps=outer)
    _, state = env.reset(jax.random.PRNGKey(2))
    step = jax.jit(env.step)
    rule0 = tuple(onp.asarray(state.rule_encoding).ravel().tolist())

    key = jax.random.PRNGKey(21)
    trials_seen, done_at = set(), None
    for t in range(inner * outer):
        key, ka, ks = jax.random.split(key, 3)
        acts = [env.action_space(a).sample(jax.random.fold_in(ka, i))
                for i, a in enumerate(env.agents)]
        _, state, _, done, _ = step(ks, state, acts)
        trials_seen.add(int(state.outer_t))
        if bool(done["__all__"]):
            done_at = t + 1
            break
        assert tuple(onp.asarray(state.rule_encoding).ravel().tolist()) == rule0, (
            f"step {t}: the hidden rule changed at a trial boundary "
            f"(outer_t={int(state.outer_t)}) — trials must share one rule")

    assert done_at == inner * outer, (
        f"meta-episode ended at step {done_at}, expected {inner * outer} "
        f"({outer} trials x {inner} steps)")
    assert len(trials_seen) >= outer - 1, (
        f"only saw trials {sorted(trials_seen)} — trial boundaries not firing")
    print(f"ok: rule held across {len(trials_seen)} trials, meta-episode ended "
          f"after {done_at} steps")


def test_ruleset_encoding_is_xland_shaped():
    """The XLand trio: fixed-shape encodings, EmptyRule zero-padding that is
    a genuine no-op, and an interpreter (check_clean) that agrees with the
    dense table the encoding came from. If any of these drift, benchmarks
    stop being interchangeable arrays and the whole
    sample/split/vmap machinery silently changes meaning."""
    # encode -> decode roundtrip over the whole space (or a 100k sample)
    tables, _full = _space_rules()
    encs = onp.asarray(jax.vmap(encode_tool_table)(jnp.asarray(tables)))
    assert encs.shape == (tables.shape[0], MAX_RULES, RULE_ENCODING_LEN), encs.shape
    assert encs.dtype == onp.uint8, encs.dtype
    back = onp.asarray(jax.vmap(
        lambda e: decode_tool_table(e, NUM_DIRT_TYPES))(jnp.asarray(encs)))
    assert (back == tables).all(), "encode_tool_table/decode_tool_table do not roundtrip"

    # padding rows are EmptyRule (all zeros) and never fire
    pad = encs[:, NUM_DIRT_TYPES:, :]
    assert (pad == EMPTY).all(), "padding rows are not zero — EmptyRule sentinel broken"
    body = encs[:, :NUM_DIRT_TYPES, 0]
    assert (body == TOOL_CLEARS).all(), "table rows are not ToolClearsRule"

    # the interpreter agrees with the table for every (waste, tool) pair
    enc0 = jnp.asarray(encs[123])
    table0 = tables[123]
    for k in range(NUM_DIRT_TYPES):
        for t in range(NUM_TOOLS):
            got = bool(check_clean(enc0, jnp.asarray(k), jnp.asarray(t)))
            assert got == (table0[k] == t), (
                f"check_clean({k}, {t}) = {got}, table says {table0[k] == t}")

    # sample_ruleset draws only encodings the enumeration contains (only
    # checkable when the space was actually enumerated)
    if _full:
        space = {e.tobytes() for e in encs}
        for s in range(50):
            draw = onp.asarray(sample_ruleset(jax.random.PRNGKey(s)))
            assert draw.tobytes() in space, "sample_ruleset left the enumerated space"

    # filter_split cuts by meaning and the halves are disjoint and exhaustive
    bench = enumerate_benchmark(jnp.asarray(encs))
    assert int(bench.num_rules[0]) == NUM_DIRT_TYPES
    waste, tool = 0, 2
    train, test = bench.filter_split(
        lambda e: decode_tool_table(e, NUM_DIRT_TYPES)[waste] != tool)
    assert train.size + test.size == bench.size
    tr = onp.asarray(jax.vmap(lambda e: decode_tool_table(e, NUM_DIRT_TYPES))(train.rules))
    te = onp.asarray(jax.vmap(lambda e: decode_tool_table(e, NUM_DIRT_TYPES))(test.rules))
    assert (tr[:, waste] != tool).all(), "a held-out pairing leaked into train"
    assert (te[:, waste] == tool).all(), "a training pairing leaked into test"
    assert abs(test.size - bench.size / NUM_TOOLS) <= 0.03 * bench.size, (train.size, test.size)
    print(f"ok: {bench.size} rulesets encode/decode/interpret consistently; "
          f"filter_split holds out {test.size} semantically unseen rulesets")


def test_reveal_rule_appends_readable_oracle_channels():
    """The oracle-input control: with reveal_rule=True the observation grows
    by NUM_DIRT_TYPES x NUM_TOOLS one-hot channels that decode back to the
    hidden rule, for every agent; with the default False the observation is
    exactly its usual shape. If the revealed block ever drifts or leaks into
    the default mode, every score in the repo changes meaning."""
    plain = opensocialjax.make("open_cleanup")
    oracle = opensocialjax.make("open_cleanup", reveal_rule=True)
    base_c = plain.observation_space()[1][-1]
    assert oracle.observation_space()[1][-1] == base_c + NUM_DIRT_TYPES * NUM_TOOLS

    obs, _ = plain.reset(jax.random.PRNGKey(9))
    assert obs[plain.agents[0]].shape[-1] == base_c, "default obs grew — leak"

    obs, st = oracle.reset(jax.random.PRNGKey(9))
    table = onp.asarray(rule_table(st))
    for a in oracle.agents:
        block = onp.asarray(obs[a])[..., base_c:]
        assert block.shape[-1] == NUM_DIRT_TYPES * NUM_TOOLS
        # identical in every cell, and argmax per waste type is the rule
        flat = block.reshape(-1, NUM_DIRT_TYPES * NUM_TOOLS)
        assert (flat == flat[0]).all(), "revealed rule varies across cells"
        decoded = flat[0].reshape(NUM_DIRT_TYPES, NUM_TOOLS).argmax(-1)
        assert (decoded == table).all(), (
            f"revealed channels decode to {decoded.tolist()}, "
            f"rule is {table.tolist()}")
    print(f"ok: reveal_rule appends {NUM_DIRT_TYPES * NUM_TOOLS} channels that "
          f"decode to the rule for all {oracle.num_agents} agents; "
          f"default obs untouched ({base_c} channels)")


def test_cleaning_chains_transform_then_clear():
    """chain_depth=2: a chain colour's tool TRANSFORMS the cell into a direct
    colour (XLand's production rule) and that colour's tool then clears it;
    chains are two hits long, end in water, and never dead-end."""
    # generator invariants over many draws
    keys = jax.random.split(jax.random.PRNGKey(0), 500)
    encs = onp.asarray(jax.vmap(lambda k: sample_chain_ruleset(k, 2, 0.5))(keys))
    depths = onp.asarray(jax.vmap(lambda e: chain_depth_of(e, NUM_DIRT_TYPES))(jnp.asarray(encs)))
    assert depths.min() >= 1 and depths.max() == 2, depths.max()
    assert (depths == 2).mean() > 0.3, "chains too rare"
    for e in encs[:50]:
        tools, nexts = map(onp.asarray, decode_chain(jnp.asarray(e), NUM_DIRT_TYPES))
        assert (tools >= 0).all(), "a colour with no rule"
        direct = nexts < 0
        assert direct.any(), "no direct colour — chains would dead-end"
        assert direct[nexts[~direct]].all(), "a chain points at a non-direct colour"
    uniq = len({e.tobytes() for e in encs})
    assert uniq > 450, f"sampled rulesets barely vary: {uniq}/500"

    # mechanism: transform, then clear
    env = opensocialjax.make("open_cleanup", chain_depth=2, p_chain=0.9)
    for seed in range(6):
        _, base = env.reset(jax.random.PRNGKey(100 + seed))
        tools = onp.asarray(rule_table(base)); nexts = onp.asarray(rule_next(base))
        chain_k = onp.flatnonzero(nexts >= 0)
        labels = onp.asarray(base.potential_dirt_and_dirt_label)
        locs = onp.asarray(base.potential_dirt_and_dirt_locs)
        hit = False
        for k in chain_k:
            idx = onp.flatnonzero(labels == int(DIRT_TYPES[k]))
            if idx.size == 0:
                continue
            target = locs[idx[0]]
            step = jax.jit(env.step)
            done_transform = False
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                r, c = int(target[0]) + dr, int(target[1]) + dc
                if not (0 <= r < env.GRID_SIZE_ROW and 0 <= c < env.GRID_SIZE_COL):
                    continue
                for facing in range(4):
                    st = place_agent(base, env, r, c, facing, int(tools[k]))
                    acts = [int(Actions.zap_clean)] + [int(Actions.stay)] * (env.num_agents - 1)
                    _, st2, _, _, _ = step(jax.random.PRNGKey(0), st, acts)
                    cell = int(onp.asarray(st2.grid)[int(target[0]), int(target[1])])
                    if cell == int(DIRT_TYPES[int(nexts[k])]):
                        done_transform = True
                        # second hit with the direct colour's tool clears it
                        st3 = place_agent(st2, env, r, c, facing, int(tools[int(nexts[k])]))
                        _, st4, _, _, _ = step(jax.random.PRNGKey(1), st3, acts)
                        cell2 = int(onp.asarray(st4.grid)[int(target[0]), int(target[1])])
                        assert cell2 == int(Items.potential_dirt), (
                            f"chain colour {k} -> {int(nexts[k])} transformed but the "
                            f"direct tool did not clear it (cell={cell2})")
                        break
                if done_transform:
                    break
            if done_transform:
                hit = True
                print(f"ok: chain colour {k} --tool {int(tools[k])}--> colour "
                      f"{int(nexts[k])} --tool {int(tools[int(nexts[k])])}--> water")
                break
        if hit:
            return
    raise AssertionError("no chain colour could be transformed from any adjacent cell")


def test_bijective_rules_are_permutations_with_all_tools_placed():
    """bijective_rules=True: every colour has its own tool and every tool
    clears exactly one colour; all NUM_DIRT_TYPES tools sit on stations; the
    matching tool clears and a mismatched one does nothing."""
    keys = jax.random.split(jax.random.PRNGKey(3), 200)
    encs = jax.vmap(sample_bijective_ruleset)(keys)
    tables = onp.asarray(jax.vmap(lambda e: decode_tool_table(e, NUM_DIRT_TYPES))(encs))
    for t in tables:
        assert sorted(t.tolist()) == list(range(NUM_DIRT_TYPES)), f"not a permutation: {t}"
    assert len({t.tobytes() for t in tables}) > 190

    try:
        env = opensocialjax.make("open_cleanup", bijective_rules=True)
    except ValueError as e:
        print(f"skip: bijective variant unavailable at this colour count ({e})")
        return
    assert env.num_tools == NUM_DIRT_TYPES
    assert set(onp.asarray(env._tool_ids).tolist()) == set(range(NUM_DIRT_TYPES)), \
        "not every tool colour has a station"
    _, base = env.reset(jax.random.PRNGKey(4))
    rule = onp.asarray(rule_table(base))
    assert sorted(rule.tolist()) == list(range(NUM_DIRT_TYPES))
    labels = onp.asarray(base.potential_dirt_and_dirt_label)
    locs = onp.asarray(base.potential_dirt_and_dirt_locs)
    k = int(onp.argmax(rule))                        # the colour whose tool is 9
    idx = onp.flatnonzero(labels == int(DIRT_TYPES[k]))
    assert idx.size > 0
    target = locs[idx[0]]
    assert try_clean(env, base, target, int(rule[k])), "own tool failed to clear"
    assert not try_clean(env, base, target, (int(rule[k]) + 1) % NUM_DIRT_TYPES), \
        "a foreign tool cleared the cell"
    print(f"ok: bijective rules are permutations; tool {int(rule[k])} clears colour {k}, "
          f"tool {(int(rule[k]) + 1) % NUM_DIRT_TYPES} does not; {env.num_tools} stations kinds placed")


def test_craft_sequence_rules():
    """Crafting variant: the hidden rule is an ORDER of station pickups.
    Wrong order -> nothing; right order (sliding window) -> the agent holds
    THE tool, which clears any colour and makes stations inert; the tool is
    lost at the trial boundary while the order is kept."""
    from opensocialjax.environments.discovery_rules import all_craft_rulesets, decode_craft_sequence
    from opensocialjax.environments.open_cleanup.open_cleanup import craft_sequence
    rs = all_craft_rulesets(4, 3)
    assert rs.shape == (64, MAX_RULES, RULE_ENCODING_LEN), rs.shape
    seqs = onp.asarray(jax.vmap(lambda e: decode_craft_sequence(e, 3))(rs))
    assert len({tuple(r) for r in seqs}) == 64

    env = opensocialjax.make("open_cleanup", num_agents=1, craft_rules=True,
                             craft_tools=4, craft_len=3, num_inner_steps=40, num_outer_steps=2)
    _, base = env.reset(jax.random.PRNGKey(5))
    step = jax.jit(env.step)
    secret = [int(x) for x in onp.asarray(craft_sequence(base, 3))]
    stations, ids = onp.asarray(env.TOOLS), onp.asarray(env._tool_ids)
    assert set(ids.tolist()) == {0, 1, 2, 3}, ids
    assert not bool(onp.asarray(base.crafted)[0])

    def pick(st, t):
        """Walk agent 0 onto some station of tool t (enter from the row above)."""
        for idx in onp.where(ids == t)[0]:
            row, col = int(stations[idx][0]), int(stations[idx][1])
            st2 = place_agent(st, env, row - 1, col, 0)
            _, st2, _, _, _ = step(jax.random.PRNGKey(1), st2, [int(Actions.up)])
            loc = onp.asarray(st2.agent_locs)[0]
            if (int(loc[0]), int(loc[1])) == (row, col):
                return st2
        raise AssertionError(f"no station of tool {t} reachable")

    # a wrong order: rotate the secret (or change one tool if it is uniform)
    wrong = secret[1:] + secret[:1]
    if wrong == secret:
        wrong = [(secret[0] + 1) % 4] + secret[1:]
    st = base
    for t in wrong:
        st = pick(st, t)
    assert not bool(onp.asarray(st.crafted)[0]), f"wrong order {wrong} crafted the tool (secret {secret})"
    # the right order, on top of the wrong pickups (sliding window)
    for t in secret:
        st = pick(st, t)
    assert bool(onp.asarray(st.crafted)[0]), f"order {secret} did not craft"
    assert onp.asarray(st.pickups)[0].tolist() == secret
    crafted_state = st
    # stations are inert afterwards
    st = pick(st, (secret[-1] + 1) % 4)
    assert bool(onp.asarray(st.crafted)[0]) and onp.asarray(st.pickups)[0].tolist() == secret

    # the beam: nothing without the tool, any colour with it
    waste = onp.asarray(base.potential_dirt_and_dirt_locs)[
        onp.isin(onp.asarray(base.potential_dirt_and_dirt_label), [int(t) for t in DIRT_TYPES])]
    hits_uncrafted = sum(try_clean(env, base, waste[i], 0) for i in range(3))
    assert hits_uncrafted == 0, "an uncrafted beam cleared waste"
    hits_crafted = sum(try_clean(env, crafted_state, waste[i], 0) for i in range(3))
    assert hits_crafted == 3, f"the crafted beam cleared only {hits_crafted}/3 cells"

    # observation: [tool flag (hidden by default), pickup one-hots...]
    obs, _, _, _, _ = step(jax.random.PRNGKey(3), crafted_state, [int(Actions.stay)])
    block = onp.asarray(obs[0])[0, 0, -NUM_COLORS:]
    assert block[0] == 0 and block[1:13].sum() == 3 and block[13:].sum() == 0, block
    env_show = opensocialjax.make("open_cleanup", num_agents=1, craft_rules=True, craft_tools=4,
                                  craft_len=3, craft_show_tool=True, num_inner_steps=40, num_outer_steps=2)
    obs2, _, _, _, _ = jax.jit(env_show.step)(jax.random.PRNGKey(3), crafted_state, [int(Actions.stay)])
    assert onp.asarray(obs2[0])[0, 0, -NUM_COLORS] == 1, "craft_show_tool=True must expose the flag"
    # trial boundary: tool lost, order kept
    st = crafted_state
    for _ in range(40):
        _, st, _, _, _ = step(jax.random.PRNGKey(2), st, [int(Actions.stay)])
    assert int(st.outer_t) == 1 and not bool(onp.asarray(st.crafted)[0]), "tool survived the trial boundary"
    assert [int(x) for x in onp.asarray(craft_sequence(st, 3))] == secret, "the order changed at the trial boundary"
    assert onp.asarray(st.pickups)[0].tolist() == [-1, -1, -1]
    print(f"ok: crafting — secret order {secret}: wrong order no, right order yes, stations inert after, "
          f"beam clears any colour only when crafted, tool lost at the trial boundary; 64 orders enumerated")


def test_attr_chain_rules():
    """Attribute chain variant: colours are hue x shade; a hue table clears
    LIGHT waste, a shade table (shared across hues) lightens MID / DARK one
    shade; the two shade tools differ; per-trial skew is on the hue."""
    from opensocialjax.environments.discovery_rules import (
        NO_EFFECT, all_attr_rulesets, apply_beam, count_rules, decode_attr_tables,
        sample_attr_ruleset)
    from opensocialjax.environments.open_cleanup.open_cleanup import (
        DIRT_BASE, NUM_HUES, NUM_SHADES, attr_tables, trick_tools)
    from opensocialjax.environments.rule_benchmark import build_benchmark

    # generator: P(4,2) * 4**5 rulesets, distinct, 7 rows, shade tools distinct
    rs = all_attr_rulesets(4, NUM_HUES, NUM_SHADES)
    assert rs.shape == (12288, MAX_RULES, RULE_ENCODING_LEN), rs.shape
    assert len({r.tobytes() for r in onp.asarray(rs).reshape(12288, -1)}) == 12288
    assert (onp.asarray(count_rules(rs)) == NUM_HUES + NUM_SHADES - 1).all()
    ht, st_ = (onp.asarray(x) for x in jax.vmap(
        lambda e: decode_attr_tables(e, NUM_HUES, NUM_SHADES))(rs))
    assert (st_[:, 0] == -1).all() and (st_[:, 1] != st_[:, 2]).all()
    assert (ht >= 0).all() and (ht < 4).all()
    assert len(build_benchmark(attr=True, attr_tools=4).rules) == 12288

    # sampler: constraint holds, every ordered shade pair shows up
    keys = jax.random.split(jax.random.PRNGKey(0), 3000)
    samp = jax.vmap(lambda k: sample_attr_ruleset(k, 4, NUM_HUES, NUM_SHADES))(keys)
    _, ss = jax.vmap(lambda e: decode_attr_tables(e, NUM_HUES, NUM_SHADES))(samp)
    ss = onp.asarray(ss)
    assert (ss[:, 1] != ss[:, 2]).all() and (ss[:, 0] == -1).all()
    assert len({tuple(r) for r in ss[:, 1:].tolist()}) == 12, "not every shade pair sampled"
    assert len({r.tobytes() for r in onp.asarray(samp).reshape(3000, -1)}) > 2000

    # interpreter: apply_beam / check_clean agree with the tables on every pair
    enc = rs[1234]
    hue_tools, shade_tools = (onp.asarray(x) for x in decode_attr_tables(enc, NUM_HUES, NUM_SHADES))
    water = int(Items.potential_dirt)
    for c in range(NUM_DIRT_TYPES):
        h, sh = divmod(c, NUM_SHADES)
        for t in range(4):
            out = int(apply_beam(enc, jnp.int32(c), jnp.int32(t), water, DIRT_BASE))
            if sh == 0:
                want = water if t == hue_tools[h] else NO_EFFECT
            else:
                want = DIRT_BASE + c - 1 if t == shade_tools[sh] else NO_EFFECT
            assert out == want, (c, t, out, want)
            assert bool(check_clean(enc, jnp.int32(c), jnp.int32(t))) == (want == water)
    tools, nexts = (onp.asarray(x) for x in decode_chain(enc, NUM_DIRT_TYPES))
    cs = onp.arange(NUM_DIRT_TYPES)
    assert (tools == onp.where(cs % 3 == 0, hue_tools[cs // 3], shade_tools[cs % 3])).all()
    assert (nexts == onp.where(cs % 3 == 0, -1, cs - 1)).all()
    assert (onp.asarray(chain_depth_of(enc, NUM_DIRT_TYPES)) == cs % 3 + 1).all()

    # the environment: dark -> mid -> light -> water, three tools, wrong ones inert
    env = opensocialjax.make("open_cleanup", num_agents=1, attr_rules=True, attr_tools=4,
                             skewed_waste=False, num_inner_steps=60, num_outer_steps=2)
    assert env.num_tools == 4 and set(onp.asarray(env._tool_ids).tolist()) == {0, 1, 2, 3}
    _, base = env.reset(jax.random.PRNGKey(7))
    step = jax.jit(env.step)
    hue_tools, shade_tools = (onp.asarray(x) for x in attr_tables(base))
    table, nxt = onp.asarray(rule_table(base)), onp.asarray(rule_next(base))
    assert (table == onp.where(cs % 3 == 0, hue_tools[cs // 3], shade_tools[cs % 3])).all()
    assert (nxt == onp.where(cs % 3 == 0, -1, cs - 1)).all()

    def fire(st, r, c, facing, tool):
        st2 = place_agent(st, env, r, c, facing, tool)
        _, st2, _, _, _ = step(jax.random.PRNGKey(0), st2, [int(Actions.zap_clean)])
        return st2

    def cell(st, target):
        return int(onp.asarray(st.grid)[int(target[0]), int(target[1])])

    def find_shot(st, target, tool, want):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = int(target[0]) + dr, int(target[1]) + dc
            if not (0 <= r < env.GRID_SIZE_ROW and 0 <= c < env.GRID_SIZE_COL):
                continue
            for facing in range(4):
                st2 = fire(st, r, c, facing, tool)
                if cell(st2, target) == want:
                    return (r, c, facing), st2
        return None, None

    locs = onp.asarray(base.potential_dirt_and_dirt_locs)
    labels = onp.asarray(base.potential_dirt_and_dirt_label)
    code = lambda c: int(DIRT_TYPES[c])
    dark = [c for c in range(NUM_DIRT_TYPES) if c % 3 == 2 and (labels == code(c)).any()]
    assert len(dark) >= 2, "need dark cells of two hues"
    c0 = dark[0]
    target = locs[onp.where(labels == code(c0))[0][0]]
    pose, st1 = find_shot(base, target, shade_tools[2], code(c0 - 1))
    assert pose is not None, "the dark tool did not lighten a dark cell"
    for t in range(4):
        if t != shade_tools[2]:
            assert cell(fire(base, *pose, t), target) == code(c0), f"tool {t} acted on a dark cell"
    st2 = fire(st1, *pose, shade_tools[1])
    assert cell(st2, target) == code(c0 - 2), "the mid tool did not lighten the mid cell"
    for t in range(4):
        if t != shade_tools[1]:
            assert cell(fire(st1, *pose, t), target) == code(c0 - 1), f"tool {t} acted on a mid cell"
    st3 = fire(st2, *pose, hue_tools[c0 // 3])
    assert cell(st3, target) == water, "the hue tool did not clear the light cell"
    for t in range(4):
        if t != hue_tools[c0 // 3]:
            assert cell(fire(st2, *pose, t), target) == code(c0 - 2), f"tool {t} cleared a light cell"
    # the shade tool is shared across hues
    c1 = next(c for c in dark if c // 3 != c0 // 3)
    target1 = locs[onp.where(labels == code(c1))[0][0]]
    pose1, _ = find_shot(base, target1, shade_tools[2], code(c1 - 1))
    assert pose1 is not None, "the dark tool is not shared across hues"

    # hue skew: 80% of the waste is the dominant hue, its shades uniform
    env_s = opensocialjax.make("open_cleanup", num_agents=1, attr_rules=True, attr_tools=4,
                               skewed_waste=True, waste_skew=0.8, num_inner_steps=60, num_outer_steps=2)
    dom_share, shade_counts, doms = [], onp.zeros(NUM_SHADES), set()
    for seed in range(20):
        _, st = env_s.reset(jax.random.PRNGKey(100 + seed))
        lab = onp.asarray(st.potential_dirt_and_dirt_label)
        waste = lab[(lab >= DIRT_BASE) & (lab < DIRT_BASE + NUM_DIRT_TYPES)] - DIRT_BASE
        dom = int(st.dom_type)
        assert 0 <= dom < NUM_HUES
        doms.add(dom)
        in_dom = waste[waste // NUM_SHADES == dom]
        dom_share.append(len(in_dom) / len(waste))
        shade_counts += onp.bincount(in_dom % NUM_SHADES, minlength=NUM_SHADES)
    share = float(onp.mean(dom_share))
    assert 0.78 <= share <= 0.90, share          # 0.8 forced + 0.2 * 1/5 by chance = 0.84
    assert (shade_counts / shade_counts.sum() >= 0.25).all(), shade_counts
    assert len(doms) >= 2
    tt = onp.asarray(trick_tools(st))
    h_t, s_t = (onp.asarray(x) for x in attr_tables(st))
    assert tt.tolist() == [h_t[int(st.dom_type)], s_t[1], s_t[2]]
    # respawns follow the hue too
    env_r = opensocialjax.make("open_cleanup", num_agents=1, attr_rules=True, attr_tools=4,
                               skewed_waste=True, waste_skew=0.8, initial_dirt=0.0,
                               dirtSpawnProbability=1.0, delayStartOfDirtSpawning=0,
                               num_inner_steps=60, num_outer_steps=2)
    step_r = jax.jit(env_r.step)
    spawned, in_dom = 0, 0
    for seed in range(3):
        _, st = env_r.reset(jax.random.PRNGKey(200 + seed))
        for i in range(40):   # a fresh key per step, or the skew draw repeats
            _, st, _, _, _ = step_r(jax.random.fold_in(jax.random.PRNGKey(seed), i), st, [int(Actions.stay)])
        lab = onp.asarray(st.potential_dirt_and_dirt_label)
        waste = lab[(lab >= DIRT_BASE) & (lab < DIRT_BASE + NUM_DIRT_TYPES)] - DIRT_BASE
        spawned += len(waste)
        in_dom += int((waste // NUM_SHADES == int(st.dom_type)).sum())
    assert spawned >= 100, spawned
    assert 0.72 <= in_dom / spawned <= 0.95, in_dom / spawned

    # trial boundary: rule and held tool survive, the map is re-laid
    st = base
    for _ in range(60):
        _, st, _, _, _ = step(jax.random.PRNGKey(2), st, [int(Actions.stay)])
    assert int(st.outer_t) == 1
    assert (onp.asarray(st.rule_encoding) == onp.asarray(base.rule_encoding)).all()
    assert int(st.held_tool[0]) == int(base.held_tool[0])
    # observation layout unchanged
    obs0, _ = opensocialjax.make("open_cleanup", num_agents=1).reset(jax.random.PRNGKey(0))
    obs1, _ = env.reset(jax.random.PRNGKey(0))
    assert onp.asarray(obs0[0]).shape == onp.asarray(obs1[0]).shape
    for bad in (dict(craft_rules=True), dict(reveal_rule=True), dict(chain_depth=2)):
        try:
            opensocialjax.make("open_cleanup", num_agents=1, attr_rules=True, **bad)
        except AssertionError:
            pass
        else:
            raise AssertionError(f"attr_rules accepted {bad}")
    print(f"ok: attribute chain — 12288 rulesets, hue table {hue_tools.tolist()} + shade tools "
          f"{shade_tools[1:].tolist()}: dark -> mid -> light -> water with three tools, wrong tools inert, "
          f"shade tool shared across hues, dominant-hue share {share:.2f}, rule/tool kept across trials")


def test_attr_craft_rules():
    """Attribute-craft variant: base tools do nothing; entering stations in a
    hidden two-pickup order puts a crafted tool in the ONE hand (one per hue,
    one per shade); the next station replaces it; recipes are distinct and
    lost with the tool at the trial boundary."""
    from opensocialjax.environments.discovery_rules import (
        CRAFT_RECIPE, count_rules, decode_attr_tables, decode_craft_recipes,
        num_attr_craft_rulesets, sample_attr_craft_ruleset)
    from opensocialjax.environments.open_cleanup.open_cleanup import (
        DIRT_BASE, NUM_HUES, NUM_SHADES, attr_tables, craft_recipes)
    from opensocialjax.environments.rule_benchmark import build_benchmark

    T_, K, NP = 4, 2, NUM_HUES + NUM_SHADES - 1
    assert num_attr_craft_rulesets(T_, K) == 57_657_600
    keys = jax.random.split(jax.random.PRNGKey(0), 500)
    rs = jax.vmap(lambda k: sample_attr_craft_ruleset(k, T_, K, NUM_HUES, NUM_SHADES))(keys)
    assert (onp.asarray(count_rules(rs)) == 2 * NP).all()
    valid, prods, seqs = (onp.asarray(x) for x in jax.vmap(lambda e: decode_craft_recipes(e, K))(rs))
    assert (valid.sum(1) == NP).all()
    for i in range(500):
        v = valid[i]
        assert prods[i][v].tolist() == list(range(T_, T_ + NP))
        orders = [tuple(s) for s in seqs[i][v].tolist()]
        assert len(set(orders)) == NP and all(0 <= x < T_ for o in orders for x in o)
    ht, st_ = (onp.asarray(x) for x in jax.vmap(lambda e: decode_attr_tables(e, NUM_HUES, NUM_SHADES))(rs))
    assert (ht == onp.arange(T_, T_ + NUM_HUES)).all() and (st_[:, 1:] == onp.array([T_ + NUM_HUES, T_ + NUM_HUES + 1])).all()
    assert len({r.tobytes() for r in onp.asarray(rs).reshape(500, -1)}) == 500
    assert len(build_benchmark(attr_craft=True, craft_tools=T_, craft_len=K, bench_size=1000).rules) >= 990

    env = opensocialjax.make("open_cleanup", num_agents=1, attr_rules=True, craft_rules=True,
                             craft_tools=T_, craft_len=K, skewed_waste=False, num_inner_steps=60, num_outer_steps=2)
    assert env.num_tools == T_ and set(onp.asarray(env._tool_ids).tolist()) == set(range(T_))
    _, base = env.reset(jax.random.PRNGKey(11))
    step = jax.jit(env.step)
    products, recipes, valid = (onp.asarray(x) for x in craft_recipes(base, K))
    recipe = {int(p): tuple(int(x) for x in s) for p, s, v in zip(products, recipes, valid) if v}
    assert sorted(recipe) == list(range(T_, T_ + NP))
    assert int(base.held_tool[0]) < T_ and not bool(base.crafted[0])
    stations, ids = onp.asarray(env.TOOLS), onp.asarray(env._tool_ids)

    def pick(st, tool):
        for idx in onp.where(ids == tool)[0]:
            row, col = int(stations[idx][0]), int(stations[idx][1])
            st2 = place_agent(st, env, row - 1, col, 0)
            _, st2, _, _, _ = step(jax.random.PRNGKey(1), st2, [int(Actions.up)])
            loc = onp.asarray(st2.agent_locs)[0]
            if (int(loc[0]), int(loc[1])) == (row, col):
                return st2
        raise AssertionError(f"no station of tool {tool} reachable")

    def fresh(st):
        return st.replace(pickups=jnp.full_like(st.pickups, -1), held_tool=st.held_tool.at[0].set(jnp.int16(0)),
                          crafted=st.crafted.at[0].set(False))

    # every recipe crafts its product; the next station replaces it (one hand)
    for prod, order in recipe.items():
        st = fresh(base)
        st = pick(st, order[0])
        assert int(st.held_tool[0]) == order[0] or int(st.held_tool[0]) >= T_   # a 1-pickup window never matches a 2-pickup recipe
        st = pick(st, order[1])
        assert int(st.held_tool[0]) == prod and bool(st.crafted[0]), (prod, order, int(st.held_tool[0]))
        assert onp.asarray(st.pickups)[0].tolist() == list(order)
        other = (order[1] + 1) % T_
        st = pick(st, other)
        chained = [p for p, o in recipe.items() if o == (order[1], other)]   # the slid window may itself be a recipe
        if chained:
            assert bool(st.crafted[0]) and int(st.held_tool[0]) == chained[0]
        else:
            assert not bool(st.crafted[0]) and int(st.held_tool[0]) == other, int(st.held_tool[0])
    # a wrong order crafts nothing
    orders = set(recipe.values())
    wrong = next((a, b) for a in range(T_) for b in range(T_) if (a, b) not in orders)
    st = pick(pick(fresh(base), wrong[0]), wrong[1])
    assert not bool(st.crafted[0]) and int(st.held_tool[0]) == wrong[1]

    # the beam: crafted ids act by hue / shade, base tools never do
    water = int(Items.potential_dirt)
    labels = onp.asarray(base.potential_dirt_and_dirt_label); locs = onp.asarray(base.potential_dirt_and_dirt_locs)
    code = lambda c: int(DIRT_TYPES[c])
    cs = onp.arange(NUM_DIRT_TYPES)
    table = onp.asarray(rule_table(base))
    assert (table == onp.where(cs % 3 == 0, T_ + cs // 3, T_ + NUM_HUES + cs % 3 - 1)).all()

    def fire(st, r, c, facing, tool):
        st2 = place_agent(st, env, r, c, facing, tool)
        _, st2, _, _, _ = step(jax.random.PRNGKey(0), st2, [int(Actions.zap_clean)])
        return st2

    def cell(st, target):
        return int(onp.asarray(st.grid)[int(target[0]), int(target[1])])

    def find_shot(st, target, tool, want):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            r, c = int(target[0]) + dr, int(target[1]) + dc
            if not (0 <= r < env.GRID_SIZE_ROW and 0 <= c < env.GRID_SIZE_COL):
                continue
            for facing in range(4):
                st2 = fire(st, r, c, facing, tool)
                if cell(st2, target) == want:
                    return (r, c, facing), st2
        return None, None

    dark = next(c for c in range(NUM_DIRT_TYPES) if c % 3 == 2 and (labels == code(c)).any())
    target = locs[onp.where(labels == code(dark))[0][0]]
    pose, st1 = find_shot(base, target, T_ + NUM_HUES + 1, code(dark - 1))
    assert pose is not None, "the crafted dark tool did not lighten a dark cell"
    for t in range(T_):
        assert cell(fire(base, *pose, t), target) == code(dark), f"base tool {t} acted on waste"
    st2 = fire(st1, *pose, T_ + NUM_HUES)
    assert cell(st2, target) == code(dark - 2)
    st3 = fire(st2, *pose, T_ + dark // 3)
    assert cell(st3, target) == water
    assert cell(fire(st2, *pose, T_ + (dark // 3 + 1) % NUM_HUES), target) == code(dark - 2), "another hue's tool cleared it"

    # observation shows the pickups, never the crafted tool; trial boundary loses tool, window and keeps recipes
    prod, order = next(iter(recipe.items()))
    st = pick(pick(fresh(base), order[0]), order[1])
    obs, _, _, _, _ = step(jax.random.PRNGKey(3), st, [int(Actions.stay)])
    block = onp.asarray(obs[0])[0, 0, -NUM_COLORS:]
    assert block[0] == 0 and block[1:1 + T_ * K].sum() == K and block[1 + T_ * K:].sum() == 0, block
    for _ in range(60):
        _, st, _, _, _ = step(jax.random.PRNGKey(2), st, [int(Actions.stay)])
    assert int(st.outer_t) == 1 and not bool(st.crafted[0]) and int(st.held_tool[0]) < T_
    assert onp.asarray(st.pickups)[0].tolist() == [-1] * K
    assert (onp.asarray(st.rule_encoding) == onp.asarray(base.rule_encoding)).all()
    print(f"ok: attribute-craft — {NP} distinct 2-pickup recipes craft tools {T_}..{T_ + NP - 1} "
          f"(recipes {recipe}); base tools inert, crafted ids act by hue/shade, one hand, "
          f"tool + window lost at the trial boundary; 57,657,600 rulesets (sampled benchmark)")


def test_a_bystander_cannot_cancel_a_shot():
    """Regression (2026-09-23). Every agent's beam footprint was scattered into the grid each step, fired or
    not, and a footprint that did not fire wrote the cell's old value back. Scatter keeps one of several writes
    to the same index, so an agent merely standing where its footprint covered another agent's target could
    erase that shot. Reproduced from a logged LLM run: shooter at (3, 6) facing north with the mid tool, a
    C1 cell two ahead at (1, 6), a bystander at (2, 7) facing north whose front-left corner is (1, 6).

    Also pins report_beam: each shooter gets the rule's own verdict on each of its targets, so two agents
    that hit one cell with the right tool are both told it worked."""
    from opensocialjax.environments.open_cleanup.open_cleanup import (
        DIRT_BASE, NUM_SHADES, attr_tables)
    from opensocialjax.environments.discovery_rules import NO_EFFECT
    mini = ["FFFHFFFFHFFFFHFF", "FFHFFFFHFFFFHFFF", "  P   P    P ~  ", "             ~  ", "   P       P ~  ",
            "             ~  ", "             ~  ", "                ", "    P      P    ",
            "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB", "BBBBBBBBBBBBBBBB"]
    env = opensocialjax.make("open_cleanup", num_agents=2, attr_rules=True, craft_rules=True, craft_tools=4,
                             craft_len=2, skewed_waste=False, map_ASCII=mini, tool_row=6, explicit_pickup=True,
                             num_inner_steps=60, num_outer_steps=2, report_beam=True)
    _, st = env.reset(jax.random.PRNGKey(3))
    step = jax.jit(env.step)
    hue_tools, shade_tools = (onp.asarray(x) for x in attr_tables(st))
    mid_tool = int(shade_tools[1])
    target, c1 = (1, 6), DIRT_BASE + 0 * NUM_SHADES + 1          # a mid cell of hue 0
    NORTH = 2

    def put(state, i, row, col, facing, tool=None):
        old = onp.asarray(state.agent_locs)[i]
        grid = state.grid.at[int(old[0]), int(old[1])].set(jnp.int16(Items.empty))
        grid = grid.at[row, col].set(jnp.int16(onp.asarray(env._agents)[i]))
        v = jnp.array([row, col, facing], dtype=state.agent_locs.dtype)
        state = state.replace(grid=grid, agent_locs=state.agent_locs.at[i].set(v), reborn_locs=state.reborn_locs.at[i].set(v))
        if tool is not None:
            state = state.replace(held_tool=state.held_tool.at[i].set(jnp.int16(tool)), crafted=state.crafted.at[i].set(True))
        return state

    # the river's colours live twice -- in the grid and in potential_dirt_and_dirt_label, which the step writes
    # back from -- so the planted C1 goes into both
    locs = onp.asarray(st.potential_dirt_and_dirt_locs)
    li = [i for i in range(len(locs)) if (int(locs[i][0]), int(locs[i][1])) == target][0]
    base = st.replace(grid=st.grid.at[target].set(jnp.int16(c1)),
                      potential_dirt_and_dirt_label=st.potential_dirt_and_dirt_label.at[li].set(
                          jnp.asarray(c1, dtype=st.potential_dirt_and_dirt_label.dtype)))
    zap, stay = int(Actions.zap_clean), int(Actions.stay)

    # 1. the shooter alone: the cell goes one shade lighter
    solo = put(put(base, 0, 3, 6, NORTH, mid_tool), 1, 8, 0, NORTH)
    _, after, _, _, _ = step(jax.random.PRNGKey(1), solo, [zap, stay])
    assert int(after.grid[target]) == c1 - 1, f"mid tool alone should lighten C1, got {int(after.grid[target])}"

    # 2. the same shot with a non-firing bystander whose footprint covers the target: still lighter
    crowd = put(put(base, 0, 3, 6, NORTH, mid_tool), 1, 2, 7, NORTH)
    _, after, _, _, info = step(jax.random.PRNGKey(1), crowd, [zap, stay])
    assert int(after.grid[target]) == c1 - 1, (
        f"a bystander at (2, 7) cancelled the shot on {target}: cell is {int(after.grid[target])}, expected {c1 - 1}")
    rc, hit, before, verdict = (onp.asarray(info[k]) for k in ("beam_rc", "beam_hit", "beam_before", "beam_after"))
    k = [j for j in range(4) if tuple(rc[0, j]) == target][0]
    assert hit[0, k] and int(before[0, k]) == c1 and int(verdict[0, k]) == c1 - 1, (hit[0, k], before[0, k], verdict[0, k])
    assert not hit[1].any(), "a bystander that did not fire must report no hits"

    # 3. two shooters with the right tool on one cell: each is told it worked, and the cell changes once
    both = put(put(base, 0, 3, 6, NORTH, mid_tool), 1, 2, 7, NORTH, mid_tool)
    _, after, _, _, info = step(jax.random.PRNGKey(1), both, [zap, zap])
    assert int(after.grid[target]) == c1 - 1
    rc, verdict = onp.asarray(info["beam_rc"]), onp.asarray(info["beam_after"])
    for i in (0, 1):
        k = [j for j in range(4) if tuple(rc[i, j]) == target][0]
        assert int(verdict[i, k]) == c1 - 1, f"agent {i} was not credited for hitting {target}: {int(verdict[i, k])}"

    # 4. a tool that does not act on the cell reports NO_EFFECT for it, and leaves it alone
    wrong = int(hue_tools[1])                                      # clears light cells of hue 1 only
    miss = put(put(base, 0, 3, 6, NORTH, wrong), 1, 8, 0, NORTH)
    _, after, _, _, info = step(jax.random.PRNGKey(1), miss, [zap, stay])
    rc, verdict = onp.asarray(info["beam_rc"]), onp.asarray(info["beam_after"])
    k = [j for j in range(4) if tuple(rc[0, j]) == target][0]
    assert int(after.grid[target]) == c1 and int(verdict[0, k]) == NO_EFFECT

    # 5. report_beam is opt-in: the RL loops reshape every info leaf to one value per actor
    plain = opensocialjax.make("open_cleanup", num_agents=2, attr_rules=True, craft_rules=True, craft_tools=4,
                               craft_len=2, skewed_waste=False, map_ASCII=mini, tool_row=6, explicit_pickup=True,
                               num_inner_steps=60, num_outer_steps=2)
    _, s0 = plain.reset(jax.random.PRNGKey(3))
    _, _, _, _, info = plain.step(jax.random.PRNGKey(1), s0, [stay, stay])
    assert not any(key.startswith("beam_") for key in info), "beam report leaked into the default info"
    print("ok: a non-firing bystander no longer cancels a shot; each shooter gets the rule's own verdict; "
          "two right-tool shooters on one cell are both credited; report_beam stays out of the default info")


if __name__ == "__main__":
    failures = []
    for fn in (test_only_the_matching_tool_clears_a_waste_type,
               test_stepping_on_a_station_picks_the_tool_up_and_leaves_it_there,
               test_the_held_tool_is_visible_in_the_observation,
               test_every_waste_type_pollutes,
               test_rule_space_is_the_full_tool_assignment_space,
               test_rule_is_shared_across_trials_of_a_meta_episode,
               test_ruleset_encoding_is_xland_shaped,
               test_reveal_rule_appends_readable_oracle_channels,
               test_cleaning_chains_transform_then_clear,
               test_bijective_rules_are_permutations_with_all_tools_placed,
               test_craft_sequence_rules,
               test_attr_chain_rules,
               test_attr_craft_rules,
               test_a_bystander_cannot_cancel_a_shot):
        try:
            fn()
        except AssertionError as e:
            failures.append(f"{fn.__name__}: {e}")
            print(f"FAIL: {fn.__name__}: {e}")
    if failures:
        sys.exit(f"{len(failures)} open_cleanup check(s) failed")
    print("ALL CLEANUP_DISCOVERY CHECKS PASSED")
