# llm_policy — language models playing OpenCleanup and OpenHarvest

Each agent is an independent LLM instance. Every step it receives one text observation and replies with one JSON
object; the harness turns the reply into an environment action, steps the (shared) JAX environment once per tick,
and writes everything down.

## What an agent sees

The observation is rebuilt every step from these blocks (`open_cleanup_prompt.compose_user`,
`open_harvest_prompt.compose_user`):

| block | what it holds | who writes it |
|---|---|---|
| progress | trial, step, the agent's own tally this trial | harness |
| **RULES YOU KNOW** | a ledger of the environment's own verdicts on the agent's actions: in OpenCleanup which pick-up order it held and what each shot did; in OpenHarvest what each apple paid and the regrowth, ripening and rotting it observed | harness (facts only) |
| **YOUR NOTES** | the agent's notebook: at most 40 entries, one add / replace / delete per step; at every trial boundary the agent rewrites it into at most 12 lessons and 4 plans, which is all it carries into the next trial | the agent |
| EARLIER TRIALS | one summary line per finished trial | harness |
| OTHER AGENTS | only what the agent has seen of the others | harness |
| YOUR PLAN / LAST STEP / RECENT | its current goal, what its last action did, the last 15 steps | harness |
| NOW | the egocentric 11 × 11 window (9 cells ahead, 1 behind, 5 to each side) as a labelled grid, plus what is in view | harness |
| ACTIONS NOW | the legal actions and where each one lands | harness |

The system prompt (`SYSTEM` in each `*_prompt.py`, written out in full) states the rules of the world and the reply
format. It never states the hidden rule set and gives no tactics. Agents cannot communicate and are not told which
model controls the others. A reply that does not parse into a legal action leaves the agent in place for that step.

## Files

| file | role |
|---|---|
| `open_cleanup_prompt.py`, `open_harvest_prompt.py` | the prompts: system text, observation blocks, reply schema, the trial-boundary notebook rewrite, and the optional cooperator / defector roles (`ROLES`) |
| `open_cleanup_policy.py`, `open_harvest_policy.py` | the harness of each game: environment construction (`make_env`), the ledger, the notebook, the agent loop (`run_episode`), a scripted provider for smoke tests |
| `providers.py`, `anthropic_provider.py` | model back ends: any OpenAI-compatible chat endpoint (with JSON-schema or JSON-object output), the Anthropic API, and vLLM's thinking controls |
| `run_experiment.py` | a formal experiment: several models × several seeds under one `config.json`, resumable, with `--status`, `--submit` (SLURM) and `--watch` |
| `run_arena.py` | five different models in one population, seats rotated per seed |
| `run_schelling.py` | one model in every seat, *k* of them prompted as cooperators and the rest as defectors |
| `replay_gif.py`, `record.py`, `observe.py` | rendering a finished run as a GIF, and shared observation helpers |

## Running

```bash
export PYTHONPATH=$PWD:$PYTHONPATH
python llm_policy/run_experiment.py --exp experiments/open_harvest_5ag --model gpt-6-luna --seed 906
python llm_policy/run_experiment.py --exp experiments/open_harvest_5ag --status
```

A run lives in `experiments/<exp>/runs/<model>/seed<k>/`:

| file | contents |
|---|---|
| `calls_agent<i>.jsonl` | every model reply for agent *i*, fsync'd the moment it arrives, with a hash of the prompt it answered |
| `decisions.jsonl` | the full step log: prompt, parsed reply, environment outcome |
| `progress.json`, `result.json`, `DONE` | where the run is; scores, notebook statistics, token usage and cost; finished |

The environment is deterministic given the seed and the actions, so on a restart each agent's provider first serves
its recorded replies back (no API calls) until the recording runs out, then continues live. A changed prompt makes
the replay stop rather than silently mix two harness versions.

**Model specs** in `config.json`:

```json
"gpt-6-luna":  {"kind": "api", "model": "gpt-6-luna", "base_url": "https://api.openai.com/v1", "key_file": "~/.config/openai/key",
                "schema_mode": "json_schema", "token_param": "max_completion_tokens", "max_tokens": 32000},
"claude":      {"kind": "anthropic", "model": "claude-opus-5-5", "key_file": "~/.config/anthropic/key", "effort": "medium", "max_tokens": 16000},
"qwen3.8-27b": {"kind": "vllm", "model_dir": "models/Qwen3.8-27B", "venv": "~/venvs/vllm", "gpus": 2, "think_effort": "medium",
                "vllm_extra": "--reasoning-parser qwen3 --tensor-parallel-size 2", "max_tokens": 6000}
```

`kind: vllm` models need a server: `scripts/slurm/exp_run.sh` starts one per job from the spec and passes
`--base-url`; to use a server you started yourself, pass `--base-url http://host:port/v1` to the runner directly.
Optional spec keys: `temperature`, `reasoning_effort`, `sampling` (extra vLLM sampling parameters), `think_budget`
(a hard cap on thinking tokens for models without effort levels), `price` (per-million-token prices, for the cost in
`result.json`), `seeds` (a model's own seed list, e.g. `["906", "906r1"]`: a replicate `906r1` replays seed 906's
problem under other run randomness), `offpeak_only` / `peak_utc` (pause live calls in a provider's peak-price hours).

**Setting keys** that change the protocol: `reveal_orders` (OpenCleanup) / `reveal_rules` (OpenHarvest) hand every
agent the hidden rules at the start; `stop_after_trials: 1` ends a run after its first trial while keeping the
prompt of the full run; `random_layout` (OpenCleanup) draws a held-out map per seed. Rewards are individual: a
setting with `"reward": "common"` is refused.
