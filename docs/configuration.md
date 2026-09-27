# Configuration reference

Every experiment is a folder under `experiments/` with a `config.json`:

```json
{
  "name": "open_harvest_5ag",
  "game": "harvest",           // "cleanup" or "harvest"
  "setting": { ... },          // the environment and the protocol, shared by every model
  "models":  { ... }           // one entry per model: how to call it and how it decodes
}
```

The eight shipped folders are the paper's settings; `experiments/demo` is a two-minute version for trying things out.

## `setting`

| key | meaning |
|---|---|
| `agents`, `inner`, `outer` | agents per population, steps per trial, trials per meta-episode (400 × 5 for OpenCleanup, 300 × 5 for OpenHarvest in the paper) |
| `seeds` | the problems: a seed fixes the map, the hidden rules and the colours, so every model faces the same ones |
| `history` | how many recent steps the observation shows (15) |
| `max_tokens` | default reply budget; a model's own `max_tokens` overrides it |
| OpenCleanup: `random_layout`, `skew`, `spawn`, `tools` | a held-out map per seed; one waste hue per meta-episode (`skew: 1.0`); waste re-appearance probability; `"shared"` tool structure |
| OpenHarvest: `pool`, `respawn_wait`, `ripen_range`, `rot_range`, `rates`, `early_stop` | the map pool (`"orig6"`); steps a zapped agent is frozen; steps per ripeness stage; regrowth delay of a rotted cell; the regrowth multipliers; fast-forward once the orchard is dead |
| `reveal_orders` (OpenCleanup) / `reveal_rules`, `reveal_regrowth` (OpenHarvest) | the rule-reveal ablation: hand every agent the hidden rules at the start |
| `stop_after_trials` | end every run after its first *k* trials while keeping the prompt of the full run |

## `models`

Every entry has a `kind` and the keys of that kind, plus optional keys any kind accepts.

```json
"gpt-6-luna": {
  "kind": "api", "model": "gpt-6-luna",
  "base_url": "https://api.openai.com/v1", "key_file": "~/.config/openai/key",
  "schema_mode": "json_schema", "token_param": "max_completion_tokens", "max_tokens": 32000,
  "price": {"input": 0.25, "cached": 0.025, "output": 2.0}
},
"claude-opus-5.5": {
  "kind": "anthropic", "model": "claude-opus-5-5", "key_file": "~/.config/anthropic/key",
  "effort": "medium", "max_tokens": 16000,
  "price": {"input": 4.0, "cached": 0.2, "output": 20.0, "cache_write": 5.0, "input_includes_cached": false}
},
"deepseek-v4.1-flash-nothink": {
  "kind": "api", "model": "deepseek-flash", "base_url": "https://api.deepseek.com/v1", "key_file": "~/.config/deepseek/key",
  "schema_mode": "json_object", "reasoning_effort": "none", "temperature": 0.2, "max_tokens": 32000,
  "offpeak_only": true, "peak_utc": [[1, 4], [6, 10]]
},
"qwen3.8-27b": {
  "kind": "vllm", "model_dir": "models/Qwen3.8-27B", "venv": "~/venvs/vllm", "gpus": 2, "serve": "legacy",
  "vllm_extra": "--reasoning-parser qwen3 --tensor-parallel-size 2",
  "think_effort": "medium", "temperature": 0.2, "max_tokens": 6000
}
```

| `kind` | keys | notes |
|---|---|---|
| `api` — any OpenAI-compatible chat endpoint | `model`, `base_url`, `key_file`; `schema_mode` (`json_schema` = strict structured output, `json_object`, or `none` when the endpoint supports neither); `token_param` (`max_tokens`, or `max_completion_tokens` for OpenAI reasoning models); `reasoning_effort`; `temperature`; `extra_body` | reasoning models take no `temperature`; leave it out to use the provider's default |
| `anthropic` — the Anthropic API | `model`, `key_file`, `effort` (`low` / `medium` / `high`) | replies are parsed as JSON without structured output |
| `vllm` — a local open-weight model | `model_dir`, `venv` (the Python environment with vLLM), `gpus`, `serve` (`legacy`, or `open` for a vLLM 0.29 build that needs the cache variables), `vllm_extra` (extra `vllm serve` flags: reasoning parser, tensor parallelism, speculative decoding, ...); `think_effort` (models with effort levels), `think_budget` (a hard cap on thinking tokens for models without them; the harness closes the think block and asks for the answer), `temperature`, `sampling` (any extra sampling parameters, e.g. `{"top_p": 0.95, "top_k": 20, "presence_penalty": 1.5}`) | the SLURM launcher starts the server and waits for it (progress in `<run>/vllm.log`); `serve: open` needs a CUDA toolkit on `PATH`, and `OSJ_CACHE` points vLLM's compile caches at a disk with room; to use a server you started yourself, pass `--base-url http://host:port/v1` to the runner |
| `scripted` | — | a fixed policy, for smoke tests without any model |

Optional keys for any kind:

| key | meaning |
|---|---|
| `max_tokens` | this model's reply budget (overrides the setting's) |
| `price` | dollars per million tokens: `input`, `cached`, `output`, `cache_write`; set `input_includes_cached: false` when the provider reports cached tokens separately. The cost lands in `result.json` |
| `seeds` | this model's own seed list, e.g. `[906, 907, "906r1"]`: the replicate `906r1` replays seed 906's problem under other run randomness |
| `time` | SLURM wall time for `--submit` (default 24 h) |
| `paused` | hold a model: `--submit` and `--watch` skip it |
| `offpeak_only`, `peak_utc` | pause live calls in the provider's peak-price hours (UTC hour ranges) |
| `thinking`, `note`, ... | free text, kept for the record |

Keys are plain text files: a `key_file` holds one key and nothing else. Nothing in the repository reads keys from
the environment.

## The three runners

| runner | config keys | what it does |
|---|---|---|
| `run_experiment.py` | `models` | homogeneous populations: every seat is the same model, one run per (model, seed); `--model M --seed S`, `--submit`, `--status`, `--watch` |
| `run_arena.py` | `roster` (five model names from `models`), `rotation: "cyclic"` | five different models in one population, one per seat, seating shifted by one per seed; tokens and cost booked per model; runs live in `runs/arena/seed<k>/` with a `seats.json` |
| `run_schelling.py` | `model` (one name from `models`), `compositions` (the values of *k*) | one model in every seat; *k* agents are prompted as cooperators (prompt item 1.5 `YOUR ROLE`, texts in `ROLES` of the prompt modules) and the rest as defectors; runs live in `runs/k<k>/seed<s>/` with a `roles.json` |

Rate limits and transient errors are retried with back-off. A run that dies anyway (a job limit, a lost connection) is
resumed from its recorded replies by the next `--watch` or by running the same command again; `offpeak_only` /
`peak_utc` pause a model's live calls in its provider's peak-price hours.

## What a run writes

`experiments/<exp>/runs/<model>/seed<k>/` (or `runs/arena/seed<k>/`, `runs/k<k>/seed<s>/`):

| file | contents |
|---|---|
| `calls_agent<i>.jsonl` | every reply of agent *i*, written and fsync'd the moment it arrives, with a hash of the prompt it answered |
| `decisions.jsonl` | one line per agent per step: the full prompt the agent saw, its parsed reply, the environment's outcome |
| `system.txt` | the system prompt of agent A (`system_cooperator.txt` / `system_defector.txt` in Schelling runs) |
| `progress.json` | trial and step reached, updated every 10 steps |
| `result.json` | scores per trial and per agent, notebook statistics, token usage and cost (`by_model` in arena runs) |
| `DONE` | present once `result.json` is complete |
| `gifs/` | written by `scripts/render_run.py` |

The environment is deterministic given the seed and the actions, so on a restart each agent's provider first serves
its recorded replies back (no model call), which rebuilds the world, the ledgers and the notebooks exactly, and the run
continues live from the first unrecorded step. Each recorded reply carries a hash of the prompt it answered; a changed
prompt makes the replay stop with `ReplayDivergence` rather than silently mix two harness versions: put the
prompt or the setting back to resume the run, or delete the run's folder to start it over.
