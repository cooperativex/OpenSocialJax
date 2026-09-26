#!/bin/bash
# One (k, seed) of a Schelling-diagram experiment (an API model in every seat). Submitted by
# `llm_policy/run_schelling.py --submit`; resubmitting the same run resumes it from its recorded replies.
#   usage: schelling_run.sh <EXP_DIR> <K> <SEED>
set -uo pipefail
EXP=${1:?exp dir}; K=${2:?k}; SEED=${3:?seed}
REPO=$(cd "$(dirname "$0")/../.." && pwd); cd $REPO || exit 1; ulimit -c 0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-OpenSocialJax}"
export PYTHONPATH=$REPO JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false
RUN=$EXP/runs/k$K/seed$SEED; mkdir -p $RUN
echo "[schelling] k=$K seed $SEED host=$(hostname) $(date -u)"
BEFORE=$(cat $RUN/calls_agent*.jsonl 2>/dev/null | wc -l)
python llm_policy/run_schelling.py --exp $EXP --k $K --seed $SEED >> $RUN/run.log 2>&1
RC=$?
AFTER=$(cat $RUN/calls_agent*.jsonl 2>/dev/null | wc -l)
echo "[schelling] exit $RC $(date -u); recorded replies $BEFORE -> $AFTER"
# FAILS counts failures that made NO progress, in a row; the watchdog stops resubmitting a run after 5.
if [ -f $RUN/DONE ] || [ $AFTER -gt $BEFORE ]; then rm -f $RUN/FAILS; fi
if [ ! -f $RUN/DONE ] && [ $RC -ne 0 ]; then echo "$(date -u) exit $RC job ${SLURM_JOB_ID:-} replies $BEFORE->$AFTER" >> $RUN/FAILS; fi
