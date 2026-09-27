"""Render a run of any runner (run_experiment, run_arena, run_schelling) as one GIF per trial, by replaying it.

The environment is deterministic given the seed and the actions, and every model reply of a run is recorded, so the
run is replayed from its recorded replies with no model call at all; the frames are drawn from the replayed states.
A run that is still going (or was interrupted) is rendered up to its last recorded step.

    python scripts/render_run.py experiments/demo/runs/scripted/seed906
    python scripts/render_run.py experiments/open_harvest_5ag/runs/gpt-6-luna/seed906 --trials 1,5 --every 2 --fps 8
    python scripts/render_run.py experiments/open_harvest_arena/runs/arena/seed906 --trials 1
    python scripts/render_run.py experiments/open_harvest_schelling/runs/k3/seed906 --trials 1

Trials are numbered from 1, as in the paper. Writes <run>/gifs/<name>_trial<k>.gif (or --out DIR); --every N keeps every
N-th step (plus every step a beam fires)."""
import argparse, json, os, sys, tempfile

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from llm_policy import jax_defaults; jax_defaults.apply()      # CPU JAX for the harness, before JAX is imported
os.chdir(REPO)


class Unrecorded(RuntimeError):
    pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="the run folder, e.g. experiments/demo/runs/scripted/seed906")
    ap.add_argument("--trials", default="1", help="trials to render, comma-separated, numbered from 1 (default: the first)")
    ap.add_argument("--every", type=int, default=2, help="keep every N-th step (default 2)")
    ap.add_argument("--fps", type=int, default=8)
    ap.add_argument("--out", default=None, help="output folder (default <run>/gifs)")
    a = ap.parse_args()

    import jax
    from llm_policy.record import GifRecorder
    from llm_policy.run_experiment import RecordingProvider, harness_args, harvest_args, load_config, make_inner_provider, stop_steps

    run = os.path.normpath(a.run); leaf = os.path.basename(run); parent = os.path.basename(os.path.dirname(run))
    exp = os.path.dirname(os.path.dirname(os.path.dirname(run)))
    cfg = load_config(exp); game = cfg.get("game", "cleanup"); setting = cfg["setting"]
    rc = json.load(open(os.path.join(run, "run_config.json")))
    n = setting["agents"]

    # which model spec (and, for the Schelling runs, which role) sits in each seat
    if os.path.exists(os.path.join(run, "seats.json")):                  # run_arena: one model per seat
        seats = rc["seats"]; specs = [cfg["models"][m] for m in seats]; roles = [None] * n; name = f"arena_{leaf}"
    elif os.path.exists(os.path.join(run, "roles.json")):                # run_schelling: one model, a role per seat
        roles = rc["roles"]; specs = [cfg["models"][cfg["model"]]] * n; name = f"{parent}_{leaf}"
    else:                                                                # run_experiment
        roles = [None] * n; specs = [cfg["models"][parent]] * n; name = f"{parent}_{leaf}"

    def never_live():
        raise Unrecorded("the run has steps with no recorded reply: only the recorded part is rendered")

    providers = [RecordingProvider(never_live, os.path.join(run, f"calls_agent{i}.jsonl")) for i in range(n)]
    if any(p.n_recorded == 0 for p in providers):
        raise SystemExit("no recorded replies in this run")
    out = a.out or os.path.join(run, "gifs"); trials = [int(x) - 1 for x in a.trials.split(",")]        # 0-based inside
    if min(trials) < 0:
        raise SystemExit("--trials are numbered from 1 (the first trial is 1)")
    log = tempfile.NamedTemporaryFile("w", suffix=".jsonl", delete=False)                  # the replay's own step log

    if game == "harvest":
        from llm_policy import open_harvest_policy as H, open_harvest_prompt as HP
        h = harvest_args(setting); ints = lambda x: [int(v) for v in str(x).split(",")]
        env = H.make_env(h.agents, h.inner, h.outer, h.ripen, h.on_train, rc["layout"], h.respawn_wait, ripen_range=ints(h.ripen_range),
                         rot_range=ints(h.rot_range), rates=[float(v) for v in str(h.rates).split(",")])
        seed = int(rc["seed"]); hues = H.present_hues(H.initial_state(env, seed, 0))
        agents = [H.Agent(providers[i], HP.build_system(h.agents, h.inner, h.outer, h.ripen, me=i, respawn_wait=h.respawn_wait, role=roles[i]),
                          i, h.agents, history=h.history, max_tokens=specs[i].get("max_tokens", h.max_tokens), strict=not h.lenient, hues=hues)
                  for i in range(n)]
        rec = GifRecorder(env, out, name, trials=trials, every=a.every, fps=a.fps)
        try:
            H.run_episode(env, agents, jax.random.PRNGKey(seed), h, log, 0, rec, max_steps=stop_steps(setting))
        except Unrecorded as e:
            print(f"[render] {e}")
    else:
        from llm_policy import open_cleanup_policy as C, open_cleanup_prompt as P
        c = harness_args(setting)
        env = C.make_env(c.agents, c.inner, c.outer, c.spawn, c.skew, c.on_train, apple_threshold=c.apple_threshold, tools=c.tools,
                         layouts="random" if c.random_layout else "fixed", problem=rc.get("problem"))
        agents = C.make_agents(c, providers)
        for i, ag in enumerate(agents):
            ag.system = P.build_system(c.agents, c.inner, c.outer, me=i, tools=c.tools, layout="random" if c.random_layout else "fixed", role=roles[i])
            ag.max_tokens = specs[i].get("max_tokens", c.max_tokens)
        key = rc.get("run_key", int(rc["seed"]))
        rec = GifRecorder(env, out, name, trials=trials, every=a.every, fps=a.fps)
        try:
            C.run_episode(env, agents, jax.random.PRNGKey(key), c, log, 0, rec, max_steps=stop_steps(setting))
        except Unrecorded as e:
            print(f"[render] {e}")
    if rec.frames:                                                        # a trial cut short: keep what there is
        rec.flush(rec.current or trials[0])
    log.close(); os.unlink(log.name)
    print("[render] " + (", ".join(rec.written) if rec.written else "nothing written: no frames in the requested trials"))


if __name__ == "__main__":
    main()
