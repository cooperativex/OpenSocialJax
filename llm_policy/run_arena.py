"""Run an arena -- five different models in one game, one per seat -- on Clean Up or Harvest (config "game"), resumable.

    experiments/<arena>/config.json         "setting" (copied verbatim from a formal experiment), "roster" (the 5 models),
                                            "rotation", and every roster model's provider spec (copied verbatim too)
    experiments/<arena>/runs/arena/seed<k>/ the same files as a formal run (calls_agent<i>.jsonl = seat i's replies,
                                            decisions.jsonl, progress.json, result.json, DONE), plus seats.json

Seating: seat i (agent letter A-E and its spawn slot) plays roster[(i + k) % 5], where k is the seed's index in
setting.seeds, so each seed shifts the seating by one. The environment, the prompts (one per seat, exactly as in the
formal runs; nobody is told which model the others are), record/replay resuming, the SLURM launcher and the watchdog all
work as in run_experiment.py. What differs: each seat gets its own provider and its own max_tokens from its model's
spec, and tokens and cost are booked per model (result.json "by_model"). At most one roster model may be a local vLLM
model: scripts/slurm/arena_run.sh starts its server and passes --base-url.

    python llm_policy/run_arena.py --exp experiments/open_cleanup_arena --seed 906    # run one seed here (resumes)
    python llm_policy/run_arena.py --exp experiments/open_cleanup_arena --submit      # every unfinished seed as a SLURM job
    python llm_policy/run_arena.py --exp experiments/open_cleanup_arena --status
    python llm_policy/run_arena.py --exp experiments/open_cleanup_arena --watch       # resubmit until all seeds are DONE
"""
from __future__ import annotations

import argparse, json, os, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCOUNT = [f"--account={os.environ['SLURM_ACCOUNT']}"] if os.environ.get("SLURM_ACCOUNT") else []   # sbatch account, if your queue needs one
sys.path.insert(0, REPO)
from llm_policy.run_experiment import (MAX_FAILS, RecordingProvider, cost_of, harness_args, harvest_args, load_config,
                                       make_inner_provider, wait_offpeak)

RUN = "arena"                                   # the run key under runs/ (a formal experiment has one per model)


def run_dir(exp, seed):
    return os.path.join(exp, "runs", RUN, f"seed{seed}")


def seat_models(cfg, seed):
    roster, n = cfg["roster"], cfg["setting"]["agents"]
    assert len(roster) == n, f"the roster has {len(roster)} models for {n} seats"
    assert cfg.get("rotation", "cyclic") == "cyclic", cfg.get("rotation")
    k = [int(s) for s in cfg["setting"]["seeds"]].index(int(seed))
    return [roster[(i + k) % n] for i in range(n)]


def vllm_models(cfg):
    ms = [m for m in cfg["roster"] if cfg["models"][m]["kind"] == "vllm"]
    assert len(ms) <= 1, f"one local model per arena (one vLLM server per job): {ms}"
    return ms


def seat_providers(cfg, seats, d, base_url, game):
    """One recording provider per seat, built from that seat's model spec (bound per seat, not by the loop variable)."""
    provs = []
    for i, m in enumerate(seats):
        spec = cfg["models"][m]; url = base_url if spec["kind"] == "vllm" else None
        provs.append(RecordingProvider(lambda spec=spec, url=url: make_inner_provider(spec, url, game=game),
                                       os.path.join(d, f"calls_agent{i}.jsonl"), before_live=lambda spec=spec: wait_offpeak(spec)))
    return provs


def book(cfg, seats, providers):
    """Tokens and cost per model (each model sits in exactly one seat) and in total."""
    by_model = {}
    for i, (m, p) in enumerate(zip(seats, providers)):
        u = dict(p.usage)
        by_model[m] = {"seat": i, "usage": u, "cost_usd": cost_of(u, cfg["models"][m].get("price"))}
    costs = [v["cost_usd"] for v in by_model.values() if v["cost_usd"] is not None]
    return by_model, round(sum(costs), 4)


def progress_writer(path, a, providers, t0):
    def on_step(trial, t, state, agents_, obs, ks):
        if t % 10 == 0:
            json.dump({"trial": trial + 1, "of_trials": a.outer, "step": t, "of_steps": a.inner,
                       "replayed": sum(p.n_replayed for p in providers), "live_calls": sum(p.n_live for p in providers),
                       "elapsed_s": round(time.time() - t0), "updated": time.strftime("%Y-%m-%d %H:%M:%S")},
                      open(path + ".tmp", "w")); os.replace(path + ".tmp", path)
    return on_step


def start(exp, seed, cfg, extra):
    d = run_dir(exp, seed); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "DONE")):
        print(f"[arena seed {seed}] already DONE"); return None, None
    seats = seat_models(cfg, seed)
    json.dump({"seed": seed, "seats": {chr(65 + i): m for i, m in enumerate(seats)}}, open(os.path.join(d, "seats.json"), "w"), indent=1)
    json.dump({"seed": seed, "seats": seats, "specs": {m: cfg["models"][m] for m in seats}, "setting": cfg["setting"], **extra},
              open(os.path.join(d, "run_config.json"), "w"), indent=1)
    return d, seats


def run_seed_cleanup(exp, seed, base_url=None):
    import jax
    from llm_policy import open_cleanup_policy as C
    cfg = load_config(exp); a = harness_args(cfg["setting"]); seed = int(seed)
    d, seats = start(exp, seed, cfg, {})
    if d is None:
        return
    env = C.make_env(a.agents, a.inner, a.outer, a.spawn, a.skew, a.on_train, apple_threshold=a.apple_threshold,
                     tools=a.tools, layouts="random" if a.random_layout else "fixed")
    providers = seat_providers(cfg, seats, d, base_url, "cleanup")
    agents = C.make_agents(a, providers)
    for ag, m in zip(agents, seats):                  # each model keeps the token cap of its own formal runs
        ag.max_tokens = cfg["models"][m].get("max_tokens", a.max_tokens)
    open(os.path.join(d, "system.txt"), "w").write(agents[0].system)
    print(f"[arena seed {seed}] seats {dict(zip('ABCDE', seats))}; recorded replies per seat {[p.n_recorded for p in providers]}", flush=True)
    t0 = time.time()
    with open(os.path.join(d, "decisions.jsonl"), "w") as log:
        per_agent, team, decisions = C.run_episode(env, agents, jax.random.PRNGKey(seed), a, log, 0, None,
                                                   on_step=progress_writer(os.path.join(d, "progress.json"), a, providers, t0))
    by_model, cost = book(cfg, seats, providers)
    nb = [{"notes": ag.notebook.n_notes, "edits": len(ag.notebook.edits),
           "rewrites_ok": sum(r["ok"] for r in ag.notebook.history), "rewrites_failed": sum(not r["ok"] for r in ag.notebook.history)}
          for ag in agents]
    res = {"arena": True, "seed": seed, "seats": seats, "team_apples_per_trial": team, "per_agent": per_agent, "decisions": decisions,
           "parse_failures": [ag.parse_failures for ag in agents], "notebook": nb, "by_model": by_model, "cost_usd": cost,
           "wall_seconds_this_session": round(time.time() - t0)}
    finish(d, res, seed, [round(x) for x in team], seats, per_agent)


def run_seed_harvest(exp, seed, base_url=None):
    import jax
    from llm_policy import open_harvest_policy as H, open_harvest_prompt as HP
    from opensocialjax.environments.open_harvest.layouts_orig6 import pick_layout
    cfg = load_config(exp); a = harvest_args(cfg["setting"]); seed = int(seed)
    assert cfg["setting"].get("pool", H.POOL) == H.POOL, f"the config's map pool {cfg['setting'].get('pool')} is not the harness's {H.POOL}"
    layout = pick_layout(seed, a.on_train)
    d, seats = start(exp, seed, cfg, {"pool": H.POOL, "layout": layout})
    if d is None:
        return
    ints = lambda x: [int(v) for v in str(x).split(",")]
    env = H.make_env(a.agents, a.inner, a.outer, a.ripen, a.on_train, layout, a.respawn_wait, ripen_range=ints(a.ripen_range),
                     rot_range=ints(a.rot_range), rates=[float(v) for v in str(a.rates).split(",")])
    providers = seat_providers(cfg, seats, d, base_url, "harvest")
    hues = H.present_hues(H.initial_state(env, seed, 0))
    systems = [HP.build_system(a.agents, a.inner, a.outer, a.ripen, me=i, respawn_wait=a.respawn_wait) for i in range(a.agents)]
    agents = [H.Agent(providers[i], systems[i], i, a.agents, history=a.history, max_tokens=cfg["models"][seats[i]].get("max_tokens", a.max_tokens),
                      strict=not a.lenient, hues=hues) for i in range(a.agents)]
    open(os.path.join(d, "system.txt"), "w").write(systems[0])
    print(f"[arena seed {seed}] harvest map {layout} of {H.POOL}; seats {dict(zip('ABCDE', seats))}; "
          f"recorded replies per seat {[p.n_recorded for p in providers]}", flush=True)
    t0 = time.time()
    with open(os.path.join(d, "decisions.jsonl"), "w") as log:
        per_agent, team, truth = H.run_episode(env, agents, jax.random.PRNGKey(seed), a, log, 0, None,
                                               on_step=progress_writer(os.path.join(d, "progress.json"), a, providers, t0))
    by_model, cost = book(cfg, seats, providers)
    nb = [{"notes": ag.notebook.n_notes, "edits": len(ag.notebook.edits), "lessons_now": list(ag.notebook.lessons),
           "rewrites_ok": sum(r["ok"] for r in ag.notebook.history), "rewrites_failed": sum(not r["ok"] for r in ag.notebook.history)}
          for ag in agents]
    res = {"arena": True, "seed": seed, "seats": seats, "pool": H.POOL, "layout": layout, "team_points_per_trial": team,
           "team_apples_per_trial": [sum(pa[t]["apples"] for pa in per_agent) for t in range(a.outer)], "per_agent": per_agent,
           "truth": truth, "parse_failures": [ag.parse_failures for ag in agents], "notebook": nb, "by_model": by_model,
           "cost_usd": cost, "wall_seconds_this_session": round(time.time() - t0)}
    finish(d, res, seed, [round(x, 1) for x in team], seats, per_agent)


def finish(d, res, seed, team, seats, per_agent):
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    mine = {m: round(sum(x["return"] for x in per_agent[i]), 1) for i, m in enumerate(seats)}
    print(f"[arena seed {seed}] DONE team per trial {team} | per model {mine} | cost ${res['cost_usd']}", flush=True)


def run_seed(exp, seed, base_url=None):
    if load_config(exp).get("game") == "harvest":
        return run_seed_harvest(exp, seed, base_url)
    return run_seed_cleanup(exp, seed, base_url)


# ---------------------------------------------------------------- status / submit / watch
def job_name(exp, seed):
    return f"x_{os.path.basename(os.path.normpath(exp))}_{seed}"


def status(exp):
    cfg = load_config(exp)
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j %T %M"], capture_output=True, text=True).stdout
    queued = {l.split()[0]: l.split()[1:] for l in q.splitlines() if l.strip()}
    short = lambda m: m.split("-")[0] if not m.startswith("gpt") else m.split("-")[-1]
    for s in cfg["setting"]["seeds"]:
        d = run_dir(exp, s); seats = seat_models(cfg, s); st, pr, sc = "-", "", ""
        if os.path.exists(os.path.join(d, "DONE")):
            r = json.load(open(os.path.join(d, "result.json"))); st = "DONE"
            sc = " ".join(f"{short(m)} {round(sum(x['return'] for x in r['per_agent'][i]), 1)}" for i, m in enumerate(seats)) + f" | ${r['cost_usd']}"
        elif os.path.exists(os.path.join(d, "progress.json")):
            p = json.load(open(os.path.join(d, "progress.json"))); st, pr = "partial", f"T{p['trial']}/{p['of_trials']} t{p['step']}/{p['of_steps']}"
        fails = os.path.join(d, "FAILS")
        if st != "DONE" and os.path.exists(fails):
            nf = sum(1 for _ in open(fails)); st = "GAVE UP" if nf >= MAX_FAILS else st; pr = (pr + f" fails {nf}").strip()
        if job_name(exp, s) in queued:
            st = queued[job_name(exp, s)][0].lower()
        print(f"seed {s}  {st:<9} {pr:<22} seats A-E: {', '.join(short(m) for m in seats)}   {sc}")


def submit(exp, seeds=None):
    cfg = load_config(exp); local = vllm_models(cfg)
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"], capture_output=True, text=True).stdout.split()
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    for s in (seeds or cfg["setting"]["seeds"]):
        job = job_name(exp, s); d = run_dir(exp, s)
        if os.path.exists(os.path.join(d, "DONE")):
            print(f"{job}: DONE, skipped"); continue
        if job in q:
            print(f"{job}: already queued or running, skipped"); continue
        fails = os.path.join(d, "FAILS")
        if os.path.exists(fails) and sum(1 for _ in open(fails)) >= MAX_FAILS:
            print(f"{job}: failed {MAX_FAILS} times in a row, NOT resubmitted -- see {fails} and run.log"); continue
        res = ["--gpus-per-node=%d" % cfg["models"][local[0]].get("gpus", 2), "--cpus-per-task=32", "--mem=256G"] if local \
            else ["--cpus-per-task=8", "--mem=32G"]
        cmd = ["sbatch", *ACCOUNT, f"--job-name={job}", "--time=24:00:00",
               f"--output={REPO}/logs/sbatch_%x_%j.out", *res,
               os.path.join(REPO, "scripts/slurm/arena_run.sh"), os.path.abspath(exp), str(s)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(f"{job}: {(out.stdout or out.stderr).strip()}")


def watch(exp, every=900):
    """Resubmit every unfinished seed that is neither DONE nor queued, every `every` seconds, until all are DONE."""
    cfg = load_config(exp)
    while True:
        left = [s for s in cfg["setting"]["seeds"] if not os.path.exists(os.path.join(run_dir(exp, s), "DONE"))]
        print(f"[watch] {time.strftime('%Y-%m-%d %H:%M:%S')} {len(left)} seeds not DONE", flush=True)
        if not left:
            print("[watch] all DONE"); return
        submit(exp)
        sys.stdout.flush()
        time.sleep(every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True); ap.add_argument("--seed")
    ap.add_argument("--seeds", default=None, help="comma-separated subset for --submit")
    ap.add_argument("--base-url", default=None, help="the local model's server (the launcher starts it and passes it)")
    ap.add_argument("--submit", action="store_true"); ap.add_argument("--status", action="store_true")
    ap.add_argument("--watch", action="store_true"); ap.add_argument("--every", type=int, default=900)
    a = ap.parse_args()
    if a.status:
        return status(a.exp)
    if a.watch:
        return watch(a.exp, a.every)
    if a.submit:
        return submit(a.exp, a.seeds.split(",") if a.seeds else None)
    run_seed(a.exp, a.seed, a.base_url)


if __name__ == "__main__":
    main()
