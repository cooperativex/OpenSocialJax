#!/bin/bash
#SBATCH --job-name=rl2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=rl2_%j.log
# RL^2 (recurrent PPO) training on one GPU: algorithms/train.py --algo RPPO on OpenCleanup or OpenHarvest.
#   usage (from the repository root): sbatch [--account=<account>] scripts/slurm/rl2_train.sh <open_cleanup|open_harvest> [overrides]
#   e.g.  sbatch scripts/slurm/rl2_train.sh open_harvest reward=common SEED=1 TOTAL_TIMESTEPS=3e8
# Overrides are Hydra's (see algorithms/RPPO/config/). Checkpoints go to checkpoints/rppo/, learning curves to
# evaluation/curves/, the log to rl2_<job id>.log in the directory sbatch was called from.
# Set CONDA_SH / CONDA_ENV if conda lives elsewhere.
set -uo pipefail
ENV=${1:?usage: rl2_train.sh <open_cleanup|open_harvest> [hydra overrides]}; shift
REPO=${OSJ_REPO:-${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}}   # sbatch runs a copy of this script from its spool dir
cd $REPO || exit 1; ulimit -c 0
source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}" && conda activate "${CONDA_ENV:-OpenSocialJax}"
# cuda must come first: if CUDA fails to initialise, JAX would otherwise fall back to the CPU without a word.
# cpu stays listed because jax.debug.callback (the wandb logging) runs on the host.
export PYTHONPATH=$REPO JAX_PLATFORMS=cuda,cpu
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
python -c "import jax; d = jax.devices()[0]; assert d.platform == 'gpu', d; print('[rl2] GPU', d, flush=True)" \
    || { echo "[rl2] no GPU visible, abort"; exit 1; }
echo "[rl2] $ENV $* host=$(hostname) $(date -u)"
python algorithms/train.py --algo RPPO --env $ENV "$@"
echo "[rl2] exit $? $(date -u)"
