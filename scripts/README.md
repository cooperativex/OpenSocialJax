# scripts — from finished runs to tables and figures

Run everything from the repository root with `PYTHONPATH=$PWD`. Scripts that build environments need JAX; the
figure scripts read the JSON the others write and need only matplotlib (and `python-pptx` for the editable
PowerPoint versions of the overview figures).

| script | what it does |
|---|---|
| `summarize_open_cleanup.py [EXP]` | per-model table of a finished OpenCleanup experiment: apples per seed, per-trial curve, cleaning, top-cleaner and top-eater shares, collapsed trials, invalid replies, cost |
| `summarize_open_harvest.py [EXP]` | the same for OpenHarvest, plus the normalised score between the scripted greedy and oracle references, best-stage share, orchard lifetime, zaps |
| `open_harvest_bounds.py <seed>` | the scripted references of a seed (greedy: eat the nearest apple; oracle: full information, best stage, keep ≥ 3 apples around every emptied cell); writes `experiments/open_harvest_5ag/bounds/seed<k>.txt` |
| `make_results_figures.py` | per-trial curves and per-model comparison figures of the homogeneous experiments |
| `make_schelling_figures.py --trials N` | Schelling diagrams of the cooperator / defector experiments (`--only T` for one trial, `--all-done` for every finished trial) |
| `open_harvest_notes_timeline.py` | what the agents' notebooks plan from trial to trial, next to zaps and orchard lifetime; reads the hand-coded counts in `docs/analysis/harvest_notes_coding.json` |
| `import_open_harvest_run.py` | imports a stand-alone `open_harvest_policy.py` run into an experiment as one (model, seed) |
| `make_env_mosaic.py`, `open_harvest_overview_data.py` + `make_open_harvest_overview.py`, `rule_generation_data.py` + `make_rule_generation_figure.py`, `figkit.py` | the environment figures: a mosaic of random maps, the OpenHarvest overview, and how the rule spaces are generated |
| `slurm/exp_run.sh`, `slurm/arena_run.sh`, `slurm/schelling_run.sh` | SLURM launchers used by `--submit`: one (model, seed) or one arena / Schelling run per job; they start a vLLM server for local models. Set `CONDA_SH` / `CONDA_ENV` if conda lives elsewhere, `SLURM_ACCOUNT` if your queue needs one, and `OSJ_CACHE` for vLLM's compile caches |

Figures are written to `docs/analysis/` (not tracked).
