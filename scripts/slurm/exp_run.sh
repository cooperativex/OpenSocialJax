#!/bin/bash
# One (model, seed) of a formal experiment. Submitted by `llm_policy/run_experiment.py --submit`, which sets the
# SLURM resources per model; resubmitting the same (model, seed) resumes it from its recorded replies.
#   usage: exp_run.sh <EXP_DIR> <MODEL> <SEED>
# API models run straight away. vLLM models first start a server with the settings in the experiment config:
#   serve=legacy -> the venv in the spec as it is; serve=open -> the same, plus the cache variables for a vLLM 0.29 build.
# Set CONDA_SH / CONDA_ENV if conda lives elsewhere, and SLURM_ACCOUNT before submitting if your queue needs one.
set -uo pipefail
EXP=${1:?exp dir}; MODEL=${2:?model}; SEED=${3:?seed}
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd $REPO || exit 1; ulimit -c 0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-OpenSocialJax}"
export PYTHONPATH=$REPO JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN=$EXP/runs/$MODEL/seed$SEED; mkdir -p $RUN
spec() { python -c "import json,sys; s=json.load(open('$EXP/config.json'))['models']['$MODEL']; v=s.get(sys.argv[1], ''); print(v if not isinstance(v,(dict,list)) else json.dumps(v))" "$1"; }
KIND=$(spec kind)
echo "[exp] $MODEL seed $SEED kind=$KIND host=$(hostname) $(date -u)"
BASE=""
if [ "$KIND" = "vllm" ]; then
  M=$(spec model_dir); VENV=$(spec venv); VENV=${VENV/#\~/$HOME}; SERVE=$(spec serve); EXTRA=$(spec vllm_extra)
  NGPU=$(spec gpus); GPUS=$(seq -s, 0 $((NGPU - 1)))
  PORT=$((8300 + ${SLURM_JOB_ID:-0} % 600))
  export TMPDIR=/tmp/vllm_${SLURM_JOB_ID:-$$}; mkdir -p $TMPDIR
  if [ "$SERVE" = "open" ]; then
    CACHE=${OSJ_CACHE:-$HOME/.cache}                 # compile caches of vLLM / Triton / Inductor: point OSJ_CACHE at a disk with room
    export VLLM_USE_FLASHINFER_SAMPLER=${VLLM_USE_FLASHINFER_SAMPLER:-0}
    export VLLM_CACHE_ROOT=$CACHE/vllm XDG_CACHE_HOME=$CACHE TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton
    mkdir -p $VLLM_CACHE_ROOT $TORCHINDUCTOR_CACHE_DIR $TRITON_CACHE_DIR
    CUDA_VISIBLE_DEVICES=$GPUS $VENV/bin/vllm serve "$M" --port $PORT --served-model-name m --max-model-len 32768 \
        --gpu-memory-utilization 0.92 --tensor-parallel-size $NGPU --max-num-seqs 16 --max-num-batched-tokens 8192 \
        $EXTRA > $RUN/vllm.log 2>&1 &
  else
    CUDA_VISIBLE_DEVICES=$GPUS $VENV/bin/vllm serve "$M" --port $PORT --served-model-name m --max-model-len 32768 \
        --gpu-memory-utilization 0.85 $EXTRA > $RUN/vllm.log 2>&1 &
  fi
  VPID=$!; trap 'kill $VPID 2>/dev/null' EXIT
  for t in $(seq 1 360); do
    curl -sf http://localhost:$PORT/v1/models >/dev/null && break
    kill -0 $VPID 2>/dev/null || { echo "[serve] vllm died"; tail -30 $RUN/vllm.log; echo "$(date -u) vllm died job ${SLURM_JOB_ID:-}" >> $RUN/FAILS; exit 1; }
    sleep 10
  done
  echo "[serve] up on $PORT $(date -u)"
  BASE="--base-url http://localhost:$PORT/v1"
fi
BEFORE=$(cat $RUN/calls_agent*.jsonl 2>/dev/null | wc -l)
python llm_policy/run_experiment.py --exp $EXP --model $MODEL --seed $SEED $BASE >> $RUN/run.log 2>&1
RC=$?
AFTER=$(cat $RUN/calls_agent*.jsonl 2>/dev/null | wc -l)
echo "[exp] exit $RC $(date -u); recorded replies $BEFORE -> $AFTER"
# FAILS counts failures that made NO progress, in a row; the watchdog stops resubmitting a run after 5.
# A run that recorded new replies before dying (a dropped connection, the 24 h limit) starts the count again.
if [ -f $RUN/DONE ] || [ $AFTER -gt $BEFORE ]; then rm -f $RUN/FAILS; fi
if [ ! -f $RUN/DONE ] && [ $RC -ne 0 ]; then echo "$(date -u) exit $RC job ${SLURM_JOB_ID:-} replies $BEFORE->$AFTER" >> $RUN/FAILS; fi
