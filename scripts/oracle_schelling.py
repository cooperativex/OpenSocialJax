"""Schelling-diagram data for the scripted bound policies (scripts/oracle_bounds.py): k = 0..5 cooperators, the rest defectors,
on the LLM experiments' problems (seeds 906-908, same keys). Seats as in llm_policy/run_schelling.py: seat i is a cooperator
when (i + j) % 5 < k, j = the seed's index in (906, 907, 908).

  cooperator  the seed's best cooperative plan from the bound sweep (bounds/oracle_seed<s>.json), played by the cooperators
              among themselves. Cleanup: at most that plan's number of cleaners, the cooperators nearest the stations, the
              other cooperators eat (and clean while no apple hangs, if the plan says so). Harvest: that plan's oracle.
  defector    the selfish bound policy. Cleanup: eats only, never cleans. Harvest: most points per step of walking now,
              any stage, no restraint. Nobody zaps.

k = 0 is the selfish bound and k = 5 the cooperative bound (same numbers). Writes bounds/oracle_schelling_seed<s>.json.
usage: python scripts/oracle_schelling.py cleanup|harvest [--seeds 906 907 908]
"""
import argparse, json, os, statistics as st, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import oracle_bounds as O

SEEDS = (906, 907, 908)


def seats_of(k, seed):
    j = SEEDS.index(int(seed))
    return ["C" if (i + j) % O.N_AGENTS < k else "D" for i in range(O.N_AGENTS)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("game", choices=["cleanup", "harvest"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS)); a = ap.parse_args()
    out = os.path.join(O.REPO, f"experiments/open_{a.game}_5ag/bounds")
    for seed in a.seeds:
        best = max((r for r in json.load(open(os.path.join(out, f"oracle_seed{seed}.json"))) if r["config"]["name"] == "coop"),
                   key=lambda r: r["total"])
        cfg = best["config"]
        env, _ = (O.cleanup_problem_env if a.game == "cleanup" else O.harvest_problem_env)(seed)
        rows = []
        for k in range(O.N_AGENTS + 1):
            t0 = time.time(); seats = seats_of(k, seed)
            r = (O.cleanup_run if a.game == "cleanup" else O.harvest_run)(env, seed, cfg, seats)
            r.update(k=k, config=cfg, seconds=round(time.time() - t0, 1)); rows.append(r)
            mean = [round(st.mean(t[i] for t in r["per_agent"]), 2) for i in range(O.N_AGENTS)]
            print(seed, "k", k, "".join(seats), "team total", r["total"], "per agent per trial", mean, f"({r['seconds']} s)", flush=True)
        json.dump(dict(plan=cfg, plan_total=best["total"], rows=rows), open(os.path.join(out, f"oracle_schelling_seed{seed}.json"), "w"), indent=1)
