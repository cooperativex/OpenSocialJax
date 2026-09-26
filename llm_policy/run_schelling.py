"""Schelling-diagram runs: one model in every seat, each agent told it is a cooperator or a defector (the prompt's 1.5
YOUR ROLE, open_cleanup_prompt.ROLES / open_harvest_prompt.ROLES), for every number of cooperators k = 0..5, on Clean Up or
Harvest (config "game"), resumable like run_experiment.py.

    experiments/<exp>/config.json          "setting" (copied verbatim from a formal experiment), "model" (its spec under
                                           "models", copied verbatim too), "compositions" (the k to run)
    experiments/<exp>/runs/k<k>/seed<s>/   the files of a formal run (calls_agent<i>.jsonl, decisions.jsonl, progress.json,
                                           result.json with "roles" and per-seat usage, DONE) plus roles.json

Roles: seat i (agent letter A-E and its spawn slot) is a cooperator when (i + j) % 5 < k, where j is the seed's index in
setting.seeds, so the cooperators' seats shift by one from seed to seed. Nobody is told the others' roles.

    python llm_policy/run_schelling.py --exp experiments/open_cleanup_schelling --k 2 --seed 906   # one run here
    python llm_policy/run_schelling.py --exp experiments/open_cleanup_schelling --submit             # every unfinished run
    python llm_policy/run_schelling.py --exp experiments/open_cleanup_schelling --status
    python llm_policy/run_schelling.py --exp experiments/open_cleanup_schelling --watch
"""
from __future__ import annotations

import argparse, json, os, statistics as st, subprocess, sys, time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ACCOUNT = [f"--account={os.environ['SLURM_ACCOUNT']}"] if os.environ.get("SLURM_ACCOUNT") else []   # sbatch account, if your queue needs one
sys.path.insert(0, REPO)
from llm_policy.run_experiment import MAX_FAILS, RecordingProvider, cost_of, harness_args, harvest_args, load_config, make_inner_provider, wait_offpeak
from llm_policy.run_arena import progress_writer


def run_dir(exp, k, seed):
    return os.path.join(exp, "runs", f"k{k}", f"seed{seed}")


def roles_of(cfg, k, seed):
    n = cfg["setting"]["agents"]; j = [int(s) for s in cfg["setting"]["seeds"]].index(int(seed))
    return ["cooperator" if (i + j) % n < int(k) else "defector" for i in range(n)]


def providers_for(cfg, d, n, game):
    spec = cfg["models"][cfg["model"]]
    assert spec["kind"] != "vllm", "run_schelling runs API models only"
    return [RecordingProvider(lambda: make_inner_provider(spec, None, game=game), os.path.join(d, f"calls_agent{i}.jsonl"),
                              before_live=lambda: wait_offpeak(spec)) for i in range(n)]


def book(cfg, roles, providers):
    price = cfg["models"][cfg["model"]].get("price")
    seats = [{"role": r, "usage": dict(p.usage), "cost_usd": cost_of(dict(p.usage), price)} for r, p in zip(roles, providers)]
    return seats, round(sum(s["cost_usd"] or 0 for s in seats), 4)


def start(exp, k, seed, cfg, extra):
    d = run_dir(exp, k, seed); os.makedirs(d, exist_ok=True)
    if os.path.exists(os.path.join(d, "DONE")):
        print(f"[k={k} seed {seed}] already DONE"); return None, None
    roles = roles_of(cfg, k, seed)
    json.dump({"k": int(k), "seed": seed, "roles": {chr(65 + i): r for i, r in enumerate(roles)}}, open(os.path.join(d, "roles.json"), "w"), indent=1)
    json.dump({"k": int(k), "seed": seed, "roles": roles, "model": cfg["model"], "spec": cfg["models"][cfg["model"]], "setting": cfg["setting"], **extra},
              open(os.path.join(d, "run_config.json"), "w"), indent=1)
    return d, roles


def finish(d, res, label):
    json.dump(res, open(os.path.join(d, "result.json"), "w"), indent=1)
    open(os.path.join(d, "DONE"), "w").write(time.strftime("%Y-%m-%d %H:%M:%S\n"))
    print(f"[{label}] DONE per seat {[(r[0], round(sum(x['return'] for x in pa), 1)) for r, pa in zip(res['roles'], res['per_agent'])]} "
          f"| cost ${res['cost_usd']}", flush=True)


def run_cleanup(exp, k, seed):
    import jax
    from llm_policy import open_cleanup_policy as C, open_cleanup_prompt as P
    cfg = load_config(exp); a = harness_args(cfg["setting"]); seed = int(seed)
    d, roles = start(exp, k, seed, cfg, {})
    if d is None:
        return
    env = C.make_env(a.agents, a.inner, a.outer, a.spawn, a.skew, a.on_train, apple_threshold=a.apple_threshold,
                     tools=a.tools, layouts="random" if a.random_layout else "fixed")
    providers = providers_for(cfg, d, a.agents, "cleanup")
    agents = C.make_agents(a, providers)
    for i, ag in enumerate(agents):                    # the same prompt as the formal runs, plus this seat's role
        ag.system = P.build_system(a.agents, a.inner, a.outer, me=i, tools=a.tools, layout="random" if a.random_layout else "fixed", role=roles[i])
        ag.max_tokens = cfg["models"][cfg["model"]].get("max_tokens", a.max_tokens)
    for r in ("cooperator", "defector"):
        if r in roles:
            open(os.path.join(d, f"system_{r}.txt"), "w").write(agents[roles.index(r)].system)
    print(f"[k={k} seed {seed}] roles {dict(zip('ABCDE', roles))}; recorded replies per seat {[p.n_recorded for p in providers]}", flush=True)
    t0 = time.time()
    with open(os.path.join(d, "decisions.jsonl"), "w") as log:
        per_agent, team, decisions = C.run_episode(env, agents, jax.random.PRNGKey(seed), a, log, 0, None,
                                                   on_step=progress_writer(os.path.join(d, "progress.json"), a, providers, t0))
    seats, cost = book(cfg, roles, providers)
    res = {"schelling": True, "k": int(k), "seed": seed, "model": cfg["model"], "roles": roles, "team_apples_per_trial": team,
           "per_agent": per_agent, "decisions": decisions, "parse_failures": [ag.parse_failures for ag in agents], "seats": seats,
           "cost_usd": cost, "wall_seconds_this_session": round(time.time() - t0)}
    finish(d, res, f"k={k} seed {seed}")


def run_harvest(exp, k, seed):
    import jax
    from llm_policy import open_harvest_policy as H, open_harvest_prompt as HP
    from opensocialjax.environments.open_harvest.layouts_orig6 import pick_layout
    cfg = load_config(exp); a = harvest_args(cfg["setting"]); seed = int(seed)
    assert cfg["setting"].get("pool", H.POOL) == H.POOL, f"the config's map pool {cfg['setting'].get('pool')} is not the harness's {H.POOL}"
    layout = pick_layout(seed, a.on_train)
    d, roles = start(exp, k, seed, cfg, {"pool": H.POOL, "layout": layout})
    if d is None:
        return
    ints = lambda x: [int(v) for v in str(x).split(",")]
    env = H.make_env(a.agents, a.inner, a.outer, a.ripen, a.on_train, layout, a.respawn_wait, ripen_range=ints(a.ripen_range),
                     rot_range=ints(a.rot_range), rates=[float(v) for v in str(a.rates).split(",")])
    providers = providers_for(cfg, d, a.agents, "harvest")
    hues = H.present_hues(H.initial_state(env, seed, 0))
    systems = [HP.build_system(a.agents, a.inner, a.outer, a.ripen, me=i, respawn_wait=a.respawn_wait, role=roles[i]) for i in range(a.agents)]
    agents = [H.Agent(providers[i], systems[i], i, a.agents, history=a.history, max_tokens=cfg["models"][cfg["model"]].get("max_tokens", a.max_tokens),
                      strict=not a.lenient, hues=hues) for i in range(a.agents)]
    for r in ("cooperator", "defector"):
        if r in roles:
            open(os.path.join(d, f"system_{r}.txt"), "w").write(systems[roles.index(r)])
    print(f"[k={k} seed {seed}] harvest map {layout} of {H.POOL}; roles {dict(zip('ABCDE', roles))}; "
          f"recorded replies per seat {[p.n_recorded for p in providers]}", flush=True)
    t0 = time.time()
    with open(os.path.join(d, "decisions.jsonl"), "w") as log:
        per_agent, team, truth = H.run_episode(env, agents, jax.random.PRNGKey(seed), a, log, 0, None,
                                               on_step=progress_writer(os.path.join(d, "progress.json"), a, providers, t0))
    seats, cost = book(cfg, roles, providers)
    res = {"schelling": True, "k": int(k), "seed": seed, "model": cfg["model"], "roles": roles, "pool": H.POOL, "layout": layout,
           "team_points_per_trial": team, "team_apples_per_trial": [sum(pa[t]["apples"] for pa in per_agent) for t in range(a.outer)],
           "per_agent": per_agent, "truth": truth, "parse_failures": [ag.parse_failures for ag in agents], "seats": seats,
           "cost_usd": cost, "wall_seconds_this_session": round(time.time() - t0)}
    finish(d, res, f"k={k} seed {seed}")


def run(exp, k, seed):
    return (run_harvest if load_config(exp).get("game") == "harvest" else run_cleanup)(exp, k, seed)


# ---------------------------------------------------------------- status / submit / watch
def job_name(exp, k, seed):
    return f"x_{os.path.basename(os.path.normpath(exp))}_k{k}_{seed}"


def all_runs(cfg):
    return [(k, s) for k in cfg["compositions"] for s in cfg["setting"]["seeds"]]


def status(exp):
    cfg = load_config(exp)
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j %T"], capture_output=True, text=True).stdout
    queued = {l.split()[0]: l.split()[1] for l in q.splitlines() if l.strip()}
    for k, s in all_runs(cfg):
        d = run_dir(exp, k, s); st_, pr, sc = "-", "", ""
        if os.path.exists(os.path.join(d, "DONE")):
            r = json.load(open(os.path.join(d, "result.json"))); st_ = "DONE"
            ret = [sum(x["return"] for x in pa) for pa in r["per_agent"]]
            c = [v for v, role in zip(ret, r["roles"]) if role == "cooperator"]; dd = [v for v, role in zip(ret, r["roles"]) if role == "defector"]
            sc = (f"coop {st.mean(c):.1f}" if c else "coop -") + (f" | defect {st.mean(dd):.1f}" if dd else " | defect -") + f" | ${r['cost_usd']}"
        elif os.path.exists(os.path.join(d, "progress.json")):
            p = json.load(open(os.path.join(d, "progress.json"))); st_, pr = "partial", f"T{p['trial']}/{p['of_trials']} t{p['step']}/{p['of_steps']}"
        fails = os.path.join(d, "FAILS")
        if st_ != "DONE" and os.path.exists(fails):
            nf = sum(1 for _ in open(fails)); st_ = "GAVE UP" if nf >= MAX_FAILS else st_; pr = (pr + f" fails {nf}").strip()
        if job_name(exp, k, s) in queued:
            st_ = queued[job_name(exp, k, s)].lower()
        print(f"k={k} seed {s}  {st_:<9} {pr:<22} {''.join('C' if r == 'cooperator' else 'D' for r in roles_of(cfg, k, s))}   {sc}")


def submit(exp):
    cfg = load_config(exp)
    q = subprocess.run(["squeue", "-u", os.environ.get("USER", ""), "-h", "-o", "%j"], capture_output=True, text=True).stdout.split()
    os.makedirs(os.path.join(REPO, "logs"), exist_ok=True)
    for k, s in all_runs(cfg):
        job = job_name(exp, k, s); d = run_dir(exp, k, s)
        if os.path.exists(os.path.join(d, "DONE")):
            continue
        if job in q:
            print(f"{job}: already queued or running, skipped"); continue
        fails = os.path.join(d, "FAILS")
        if os.path.exists(fails) and sum(1 for _ in open(fails)) >= MAX_FAILS:
            print(f"{job}: failed {MAX_FAILS} times in a row, NOT resubmitted -- see {fails} and run.log"); continue
        cmd = ["sbatch", *ACCOUNT, f"--job-name={job}", "--time=24:00:00", f"--output={REPO}/logs/sbatch_%x_%j.out",
               "--cpus-per-task=8", "--mem=32G", os.path.join(REPO, "scripts/slurm/schelling_run.sh"), os.path.abspath(exp), str(k), str(s)]
        out = subprocess.run(cmd, capture_output=True, text=True)
        print(f"{job}: {(out.stdout or out.stderr).strip()}")


def watch(exp, every=900):
    cfg = load_config(exp)
    while True:
        left = [(k, s) for k, s in all_runs(cfg) if not os.path.exists(os.path.join(run_dir(exp, k, s), "DONE"))]
        print(f"[watch] {time.strftime('%Y-%m-%d %H:%M:%S')} {len(left)} runs not DONE", flush=True)
        if not left:
            print("[watch] all DONE"); return
        submit(exp); sys.stdout.flush(); time.sleep(every)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--exp", required=True); ap.add_argument("--k"); ap.add_argument("--seed")
    ap.add_argument("--submit", action="store_true"); ap.add_argument("--status", action="store_true")
    ap.add_argument("--watch", action="store_true"); ap.add_argument("--every", type=int, default=900)
    a = ap.parse_args()
    if a.status:
        return status(a.exp)
    if a.watch:
        return watch(a.exp, a.every)
    if a.submit:
        return submit(a.exp)
    run(a.exp, a.k, a.seed)


if __name__ == "__main__":
    main()
