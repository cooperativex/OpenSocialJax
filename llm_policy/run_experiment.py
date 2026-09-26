"""Run a formal experiment -- Clean Up or Harvest (config "game") -- several models x several seeds, one folder, resumable.

    experiments/<name>/config.json              the setting and every model's provider settings (tracked in git)
    experiments/<name>/runs/<model>/seed<k>/
        calls_agent<i>.jsonl   every model reply for agent i, written and fsync'd the moment it arrives
        decisions.jsonl        the full step log (prompt, reply, outcome), regenerated on every (re)start
        system.txt             agent 0's system prompt
        progress.json          trial / step reached, updated as the run goes
        result.json            scores, notebook stats, token usage and cost -- written at the end
        DONE                   present once result.json is complete

Resuming. The environment is deterministic: the same seed and the same actions give the same world, and the
only thing that is not reproducible is what the model says. So every model reply is appended to
calls_agent<i>.jsonl before the harness acts on it. On a restart each agent's provider first serves those
recorded replies back, in order and without calling the API, which rebuilds the world, the ledgers and the
notebooks exactly as they were; when the recording runs out it carries on live. A crash or a dropped
connection costs at most the calls that were in flight. Each recorded reply carries a hash of the prompt it
answered, and the replay stops with an error if a prompt ever differs -- a changed harness must never be
silently mixed into a half-finished run.

    # run one seed here (resumes by itself)
    python llm_policy/run_experiment.py --exp experiments/cleanup_5ag --model gpt-6-luna --seed 906
    # submit every unfinished seed of a model as its own SLURM job (skips DONE and already-queued ones)
    python llm_policy/run_experiment.py --exp experiments/cleanup_5ag --model gpt-6-luna --submit
    # where everything stands
    python llm_policy/run_experiment.py --exp experiments/cleanup_5ag --status
"""
from __future__ import annotations

import argparse, collections, hashlib, json, os, subprocess, sys, time, types

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCOUNT = [f"--account={os.environ['SLURM_ACCOUNT']}"] if os.environ.get("SLURM_ACCOUNT") else []   # sbatch account, if your queue needs one


# ---------------------------------------------------------------- config
def load_config(exp):
    return json.load(open(os.path.join(exp, "config.json")))


def run_dir(exp, model, seed):
    return os.path.join(exp, "runs", model, f"seed{seed}")


def parse_seed(seed):
    """'906' -> (906, 0): the seed's own run. '906r2' -> (906, 2): a replicate of seed 906's problem (same map, hidden
    rule and waste hue), played with run key 906 * 1000 + 2 so that spawns, waste shades, respawns and apples differ."""
    s = str(seed)
    if "r" in s:
        p, r = s.split("r")
        return int(p), int(r)
    return int(s), 0


def harness_args(setting):
    """The namespace open_cleanup_policy's make_env / make_agents / run_episode read. Rewards are individual only; a
    config from before 2026-09-23 that says "reward": "common" is refused rather than silently run individual."""
    s = dict(setting)
    if s.get("reward", "individual") != "individual":
        raise SystemExit(f'setting says reward={s["reward"]!r}: common reward was removed (2026-09-23), rewards are individual only')
    return types.SimpleNamespace(
        agents=s["agents"], history=s["history"], inner=s["inner"], outer=s["outer"],
        spawn=s["spawn"], skew=s["skew"], on_train=s.get("on_train", False), apple_threshold=s.get("apple_threshold"),
        tools=s.get("tools", "shared"), random_layout=s.get("random_layout", False), max_tokens=s["max_tokens"],
        lenient=s.get("lenient", False), needed_hint=s.get("needed_hint", False),
        reveal_orders=s.get("reveal_orders", False), curriculum=s.get("curriculum", "none"))


def make_inner_provider(spec, base_url=None, game="cleanup"):
    kind = spec["kind"]
    if kind == "api":
        from llm_policy.providers import APIProvider
        return APIProvider(spec["model"], spec["base_url"], spec["key_file"], temperature=spec.get("temperature"),
                           reasoning_effort=spec.get("reasoning_effort"), extra_body=spec.get("extra_body"),
                           schema_mode=spec.get("schema_mode", "json_schema"), token_param=spec.get("token_param", "max_tokens"))
    if kind == "anthropic":
        from llm_policy.anthropic_provider import AnthropicProvider
        return AnthropicProvider(spec["model"], spec.get("key_file", "~/.config/anthropic/key"), effort=spec.get("effort", "low"))
    if kind == "vllm":
        from llm_policy.open_cleanup_policy import ChatProvider
        assert base_url, "a vllm model needs --base-url (the launcher starts the server and passes it)"
        extra = {"chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": spec["think_effort"]}} if spec.get("think_effort") else {}
        extra.update(spec.get("sampling", {}))   # e.g. top_p / top_k / presence_penalty, passed straight to vLLM
        return ChatProvider("m", base_url, "EMPTY", spec.get("temperature", 0.2), extra_body=extra or None, timeout=1200.0,
                            think_budget=spec.get("think_budget"))   # a hard cap on thinking, for models with no effort levels
    if kind == "scripted":
        if game == "harvest":
            from llm_policy.open_harvest_policy import ScriptedProvider as HarvestScripted
            return HarvestScripted()
        from llm_policy.open_cleanup_policy import ScriptedProvider
        return ScriptedProvider()
    raise ValueError(kind)


def harvest_args(setting):
    """The namespace open_harvest_policy's make_env / Agent / run_episode read. Rewards are individual only."""
    s = dict(setting)
    if s.get("reward", "individual") != "individual":
        raise SystemExit(f'setting says reward={s["reward"]!r}: rewards are individual only')
    return types.SimpleNamespace(
        agents=s["agents"], history=s["history"], inner=s["inner"], outer=s["outer"], ripen=s.get("ripen", 25),
        respawn_wait=s.get("respawn_wait", 12), ripen_range=s.get("ripen_range", "15,20"), rot_range=s.get("rot_range", "2,5"),
        rates=s.get("rates", "2,1,0.5"), on_train=s.get("on_train", False), max_tokens=s["max_tokens"],
        lenient=s.get("lenient", False), early_stop=s.get("early_stop", True),
        reveal_rules=s.get("reveal_rules", False), reveal_regrowth=s.get("reveal_regrowth", False))   # the rule-reveal ablation only


# ---------------------------------------------------------------- record / replay
class ReplayDivergence(RuntimeError):
    pass


def _load_calls(path):
    recs = []
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    break                          # a line cut off by the crash: that call is simply made again
    return recs


def is_peak(spec, now=None):
    """DeepSeek bills double in UTC 01:00-04:00 and 06:00-10:00, Monday to Friday. spec["peak_utc"] lists them."""
    win = spec.get("peak_utc")
    if not win:
        return False
    t = time.gmtime(now or time.time())
    return t.tm_wday < 5 and any(a <= t.tm_hour < b for a, b in win)


def wait_offpeak(spec):
    """Sleep through a peak window before a live call. Replayed calls never wait: they cost nothing."""
    if not spec.get("offpeak_only"):
        return
    waited = False
    while is_peak(spec):
        if not waited:
            print(f"[offpeak] peak pricing now ({time.strftime('%a %H:%M', time.gmtime())} UTC); pausing", flush=True); waited = True
        time.sleep(60)
    if waited:
        print(f"[offpeak] resumed {time.strftime('%a %H:%M', time.gmtime())} UTC", flush=True)


class RecordingProvider:
    """Serves the recorded replies first, then calls the real provider and records each new reply."""

    def __init__(self, make_inner, path, before_live=None):
        self.path, self.make_inner, self._inner = path, make_inner, None
        self.before_live = before_live
        recs = _load_calls(path)
        if recs and os.path.getsize(path) and not open(path, "rb").read().endswith(b"\n"):
            # drop the torn tail so the next append starts on a fresh line
            with open(path, "w") as f:
                f.writelines(json.dumps(r) + "\n" for r in recs)
        self.replay = collections.deque(recs)
        self.n_recorded, self.n_replayed, self.n_live = len(recs), 0, 0
        self.usage = collections.Counter()
        for r in recs:
            self.usage.update({k: v for k, v in (r.get("usage") or {}).items()})
            self.usage["calls"] += 1; self.usage["seconds"] += r.get("seconds") or 0
        self.model = "replay"
        self.last_reasoning, self.last_tokens, self.last_seconds, self.last_reasoning_tokens = "", 0, 0.0, 0

    @property
    def inner(self):
        if self._inner is None:
            self._inner = self.make_inner()
            self.model = getattr(self._inner, "model", "?")
        return self._inner

    @staticmethod
    def _hash(system, messages):
        return hashlib.sha1((system + "\x00" + json.dumps(messages, sort_keys=True)).encode()).hexdigest()[:16]

    def complete(self, system, messages, max_tokens=400, schema=None):
        h = self._hash(system, messages)
        if self.replay:
            rec = self.replay.popleft()
            if rec["h"] != h:
                raise ReplayDivergence(f"{self.path}: recorded call {self.n_replayed} answered a different prompt; "
                                       "the harness or the setting changed since this run started")
            self.n_replayed += 1
            self.last_reasoning, self.last_tokens = rec.get("reasoning", ""), rec.get("tokens", 0)
            self.last_seconds, self.last_reasoning_tokens = rec.get("seconds", 0.0), rec.get("reasoning_tokens", 0)
            return rec["raw"]
        if self.before_live:
            self.before_live()
        inner = self.inner
        before = dict(getattr(inner, "usage", {}))
        raw = inner.complete(system, messages, max_tokens, schema)
        after = getattr(inner, "usage", {})
        delta = {k: after[k] - before.get(k, 0) for k in after if isinstance(after[k], (int, float)) and k not in ("calls", "seconds")}
        rec = {"h": h, "raw": raw, "reasoning": getattr(inner, "last_reasoning", "") or "",
               "tokens": getattr(inner, "last_tokens", 0), "reasoning_tokens": getattr(inner, "last_reasoning_tokens", 0),
               "seconds": getattr(inner, "last_seconds", 0.0), "stop": getattr(inner, "last_stop", None),
               "usage": delta, "t": round(time.time(), 1)}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec) + "\n"); f.flush(); os.fsync(f.fileno())
        self.n_live += 1
        self.usage.update(delta); self.usage["calls"] += 1; self.usage["seconds"] += rec["seconds"] or 0
        self.last_reasoning, self.last_tokens = rec["reasoning"], rec["tokens"]
        self.last_seconds, self.last_reasoning_tokens = rec["seconds"], rec["reasoning_tokens"]
        return raw


# ---------------------------------------------------------------- one seed
def cost_of(usage, price):
    """price: $/1M for input (uncached), cached, output (and cache_write for Anthropic)."""
    if not price:
        return None
    inp, cached, out = usage.get("input_tokens", 0), usage.get("cached_tokens", 0), usage.get("output_tokens", 0)
    uncached = inp - cached if price.get("input_includes_cached", True) else inp
    return round((uncached * price["input"] + cached * price.get("cached", 0) + out * price["output"]
                  + usage.get("cache_write_tokens", 0) * price.get("cache_write", 0)) / 1e6, 4)


def stop_steps(setting):
    """setting "stop_after_trials": k ends the run after its first k trials (env steps k * inner), with outer left as it is
    so the prompt and the environment match the full runs (the rule-reveal ablation, 2026-09-25). Absent: run every trial."""
    k = setting.get("stop_after_trials")
    return None if k is None else int(k) * int(setting["inner"])


def run_seed(exp, model, seed, base_url=None):
    if parse_seed(seed)[1] == 0:
        seed = parse_seed(seed)[0]                            # a plain seed stays an int in run_config / result.json
    if load_config(exp).get("game") == "harvest":
        return run_seed_harvest(exp, model, seed, base_url)
    import jax
    from llm_policy import open_cleanup_policy as C
    cfg = load_config(exp); spec = cfg["models"][model]; a = harness_args(cfg["setting"])
    a.max_tokens = spec.get("max_tokens", a.max_tokens)       # thinking models need room for the thinking
    d = run_dir(exp, model, seed); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "DONE")):
        print(f"[{model} seed {seed}] already DONE"); return
    problem_seed, rep = parse_seed(seed)
    problem = None
    if rep:                                                   # a replicate: pin the problem seed's map, rule and hue
        assert a.random_layout, "replicates pin a map of the random layout pool"
        problem = C.problem_of(problem_seed, a.agents, a.inner, a.outer, a.spawn, a.skew, a.on_train, a.apple_threshold, a.tools)
    run_key = problem_seed * 1000 + rep if rep else problem_seed
    json.dump({"model": model, "seed": seed, "problem": problem, "run_key": run_key, "spec": spec, "setting": cfg["setting"]},
              open(os.path.join(d, "run_config.json"), "w"), indent=1)
    env = C.make_env(a.agents, a.inner, a.outer, a.spawn, a.skew, a.on_train, apple_threshold=a.apple_threshold,
                     tools=a.tools, layouts="random" if a.random_layout else "fixed", problem=problem)
    providers = [RecordingProvider(lambda: make_inner_provider(spec, base_url), os.path.join(d, f"calls_agent{i}.jsonl"),
                                   before_live=lambda: wait_offpeak(spec))
                 for i in range(a.agents)]
    agents = C.make_agents(a, providers)
    open(os.path.join(d, "system.txt"), "w").write(agents[0].system)
    recorded = [p.n_recorded for p in providers]
    print(f"[{model} seed {seed}] start; recorded replies per agent {recorded} -> replaying those, then live", flush=True)
    t0 = time.time()
    prog = os.path.join(d, "progress.json")

    def on_step(trial, t, state, agents_, obs, ks):
        if t % 10 == 0:
            live = sum(p.n_live for p in providers)
            json.dump({"trial": trial + 1, "of_trials": a.outer, "step": t, "of_steps": a.inner,
                       "replayed": sum(p.n_replayed for p in providers), "live_calls": live,
                       "elapsed_s": round(time.time() - t0), "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                      open(prog + ".tmp", "w")); os.replace(prog + ".tmp", prog)

    with open(os.path.join(d, "decisions.jsonl"), "w") as log:
        per_agent, team, decisions = C.run_episode(env, agents, jax.random.PRNGKey(run_key), a, log, 0, None, on_step=on_step,
                                                   max_steps=stop_steps(cfg["setting"]))
    usage = collections.Counter()
    for p in providers:
        usage.update(p.usage)
    usage = dict(usage)
    nb = [{"notes": ag.notebook.n_notes, "edits": len(ag.notebook.edits),
           "rewrites_ok": sum(r["ok"] for r in ag.notebook.history), "rewrites_failed": sum(not r["ok"] for r in ag.notebook.history)}
          for ag in agents]
    res = {"model": model, "seed": seed, "problem": problem, "run_key": run_key, "team_apples_per_trial": team, "per_agent": per_agent, "decisions": decisions,
           "parse_failures": [ag.parse_failures for ag in agents], "notebook": nb, "usage": usage,
           "cost_usd": cost_of(usage, spec.get("price")), "wall_seconds_this_session": round(time.time() - t0)}
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[{model} seed {seed}] DONE team apples/trial {[round(x) for x in team]} | cost ${res['cost_usd']} | "
          f"{usage.get('calls', 0)} calls", flush=True)


def run_seed_harvest(exp, model, seed, base_url=None):
    """One (model, seed) of a Harvest experiment: the harness baseline (open_harvest_policy.make_env, pool POOL, the seed's map
    from pick_layout), resumable exactly like a Clean Up run (recorded replies are replayed first)."""
    import jax
    from llm_policy import open_harvest_policy as H, open_harvest_prompt as HP
    from opensocialjax.environments.open_harvest.layouts_orig6 import pick_layout
    cfg = load_config(exp); spec = cfg["models"][model]; a = harvest_args(cfg["setting"])
    assert cfg["setting"].get("pool", H.POOL) == H.POOL, f"the config's map pool {cfg['setting'].get('pool')} is not the harness's {H.POOL}"
    a.max_tokens = spec.get("max_tokens", a.max_tokens)
    seed = int(seed)
    d = run_dir(exp, model, seed); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "DONE")):
        print(f"[{model} seed {seed}] already DONE"); return
    layout = pick_layout(seed, a.on_train)
    json.dump({"model": model, "seed": seed, "pool": H.POOL, "layout": layout, "spec": spec, "setting": cfg["setting"]},
              open(os.path.join(d, "run_config.json"), "w"), indent=1)
    ints = lambda x: [int(v) for v in str(x).split(",")]
    env = H.make_env(a.agents, a.inner, a.outer, a.ripen, a.on_train, layout, a.respawn_wait, ripen_range=ints(a.ripen_range),
                     rot_range=ints(a.rot_range), rates=[float(v) for v in str(a.rates).split(",")])
    providers = [RecordingProvider(lambda: make_inner_provider(spec, base_url, game="harvest"), os.path.join(d, f"calls_agent{i}.jsonl"),
                                   before_live=lambda: wait_offpeak(spec))
                 for i in range(a.agents)]
    hues = H.present_hues(H.initial_state(env, seed, 0))
    systems = [HP.build_system(a.agents, a.inner, a.outer, a.ripen, me=i, respawn_wait=a.respawn_wait) for i in range(a.agents)]
    agents = [H.Agent(providers[i], systems[i], i, a.agents, history=a.history, max_tokens=a.max_tokens, strict=not a.lenient, hues=hues)
              for i in range(a.agents)]
    open(os.path.join(d, "system.txt"), "w").write(systems[0])
    print(f"[{model} seed {seed}] harvest map {layout} of {H.POOL}; recorded replies per agent {[p.n_recorded for p in providers]} "
          f"-> replaying those, then live", flush=True)
    t0 = time.time(); prog = os.path.join(d, "progress.json")

    def on_step(trial, t, state, agents_, obs, ks):
        if t % 10 == 0:
            json.dump({"trial": trial + 1, "of_trials": a.outer, "step": t, "of_steps": a.inner,
                       "replayed": sum(p.n_replayed for p in providers), "live_calls": sum(p.n_live for p in providers),
                       "elapsed_s": round(time.time() - t0), "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                      open(prog + ".tmp", "w")); os.replace(prog + ".tmp", prog)

    with open(os.path.join(d, "decisions.jsonl"), "w") as log:
        per_agent, team, truth = H.run_episode(env, agents, jax.random.PRNGKey(seed), a, log, 0, None, on_step=on_step,
                                               max_steps=stop_steps(cfg["setting"]))
    usage = collections.Counter()
    for p in providers:
        usage.update(p.usage)
    usage = dict(usage)
    nb = [{"notes": ag.notebook.n_notes, "edits": len(ag.notebook.edits), "lessons_now": list(ag.notebook.lessons),
           "rewrites_ok": sum(r["ok"] for r in ag.notebook.history), "rewrites_failed": sum(not r["ok"] for r in ag.notebook.history)}
          for ag in agents]
    res = {"model": model, "seed": seed, "pool": H.POOL, "layout": layout, "team_points_per_trial": team,
           "team_apples_per_trial": [sum(pa[t]["apples"] for pa in per_agent) for t in range(a.outer)], "per_agent": per_agent,
           "truth": truth, "parse_failures": [ag.parse_failures for ag in agents], "notebook": nb, "usage": usage,
           "cost_usd": cost_of(usage, spec.get("price")), "wall_seconds_this_session": round(time.time() - t0)}
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[{model} seed {seed}] DONE team points/trial {[round(x, 1) for x in team]} | cost ${res['cost_usd']} | "
          f"{usage.get('calls', 0)} calls", flush=True)


# ---------------------------------------------------------------- status / submit
def status(exp):
    cfg = load_config(exp)
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j %T %M"], capture_output=True, text=True).stdout
    queued = {l.split()[0]: l.split()[1:] for l in q.splitlines() if l.strip()}
    print(f"{'model':<22}{'seed':>6}  {'state':<10}{'progress':<24}{'team apples/trial':<28}{'cost':>8}")
    for m in cfg["models"]:
        if cfg["models"][m].get("skip"):
            continue
        for s in model_seeds(cfg, m):
            d = run_dir(exp, m, s); job = f"x_{os.path.basename(exp)}_{m}_{s}"
            st, pr, sc, co = "-", "", "", ""
            if os.path.exists(os.path.join(d, "DONE")):
                r = json.load(open(os.path.join(d, "result.json")))
                vals = [round(x, 1) for x in r["team_points_per_trial"]] if "team_points_per_trial" in r else [round(x) for x in r["team_apples_per_trial"]]
                st, sc, co = "DONE", str(vals), f"${r['cost_usd']}" if r["cost_usd"] is not None else ""
            elif os.path.exists(os.path.join(d, "progress.json")):
                p = json.load(open(os.path.join(d, "progress.json")))
                st, pr = "partial", f"T{p['trial']}/{p['of_trials']} t{p['step']}/{p['of_steps']}"
            fails = os.path.join(d, "FAILS")
            if st != "DONE" and os.path.exists(fails):
                nf = sum(1 for _ in open(fails)); st = "GAVE UP" if nf >= MAX_FAILS else st
                pr = (pr + f" fails {nf}").strip()
            if job in queued:
                st = queued[job][0].lower()
            print(f"{m:<22}{s:>6}  {st:<10}{pr:<24}{sc:<28}{co:>8}")


def model_seeds(cfg, model):
    """The seeds a model runs: its own "seeds" list in config.json when set (e.g. one seed of an expensive model), else the setting's."""
    return cfg["models"][model].get("seeds") or cfg["setting"]["seeds"]


def submit(exp, model, seeds=None):
    cfg = load_config(exp); spec = cfg["models"][model]
    if spec.get("paused"):                       # set in config.json to hold a model; its runs resume when it is removed
        print(f"{model}: paused in config, not submitted"); return
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"], capture_output=True, text=True).stdout.split()
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    for s in (seeds or model_seeds(cfg, model)):
        job = f"x_{os.path.basename(exp)}_{model}_{s}"
        if os.path.exists(os.path.join(run_dir(exp, model, s), "DONE")):
            print(f"{job}: DONE, skipped"); continue
        if job in q:
            print(f"{job}: already queued or running, skipped"); continue
        fails = os.path.join(run_dir(exp, model, s), "FAILS")
        if os.path.exists(fails) and sum(1 for _ in open(fails)) >= MAX_FAILS:
            print(f"{job}: failed {MAX_FAILS} times in a row, NOT resubmitted -- see {fails} and run.log"); continue
        res = ["--gpus-per-node=%d" % spec.get("gpus", 2), "--cpus-per-task=32", "--mem=256G"] if spec["kind"] == "vllm" \
            else ["--cpus-per-task=8", "--mem=32G"]
        cmd = ["sbatch", *ACCOUNT, f"--job-name={job}", f"--time={spec.get('time', '24:00:00')}",
               f"--output={REPO}/logs/sbatch_%x_%j.out", *res,
               os.path.join(REPO, "scripts/slurm/exp_run.sh"), os.path.abspath(exp), model, str(s)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(f"{job}: {(out.stdout or out.stderr).strip()}")


MAX_FAILS = 5


def watch(exp, every=900):
    """Resubmit every unfinished (model, seed) that is neither DONE nor queued, every `every` seconds, until all
    are DONE. A run that dies (a dropped connection past the retries, a node failure, the 24 h queue limit) is
    back within one interval and resumes from its recorded replies; one that fails MAX_FAILS times running is
    left alone for a person to look at."""
    cfg = load_config(exp)
    models = [m for m in cfg["models"] if not cfg["models"][m].get("skip") and not cfg["models"][m].get("paused")]
    while True:
        left = [(m, s) for m in models for s in model_seeds(cfg, m) if not os.path.exists(os.path.join(run_dir(exp, m, s), "DONE"))]
        print(f"[watch] {time.strftime('%Y-%m-%d %H:%M:%S')} {len(left)} runs not DONE", flush=True)
        if not left:
            print("[watch] all DONE"); return
        for m in models:
            submit(exp, m)
        sys.stdout.flush()
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True)
    ap.add_argument("--model"); ap.add_argument("--seed", help="906, or 906r1 for a replicate of seed 906's problem")
    ap.add_argument("--seeds", default=None, help="comma-separated subset for --submit")
    ap.add_argument("--base-url", default=None, help="vllm models: the server the launcher started")
    ap.add_argument("--submit", action="store_true"); ap.add_argument("--status", action="store_true")
    ap.add_argument("--watch", action="store_true", help="loop: resubmit unfinished runs every --every seconds until all are DONE")
    ap.add_argument("--every", type=int, default=900)
    a = ap.parse_args()
    if a.status:
        return status(a.exp)
    if a.watch:
        return watch(a.exp, a.every)
    if a.submit:
        return submit(a.exp, a.model, a.seeds.split(",") if a.seeds else None)
    run_seed(a.exp, a.model, a.seed, a.base_url)


if __name__ == "__main__":
    sys.path.insert(0, REPO)
    main()
