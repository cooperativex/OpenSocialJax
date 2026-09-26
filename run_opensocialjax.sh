#!/bin/bash
#SBATCH --job-name=opensocialjax
#SBATCH --gpus=1
#SBATCH --time=24:00:00
#SBATCH --output=logs/opensocialjax_%j.out

# Usage (mirrors ../SocialJax/run_socialjax.sh):
#   sbatch run_opensocialjax.sh                            # IPPO open_cleanup (defaults)
#   sbatch run_opensocialjax.sh RPPO open_cleanup       # pick algo/env
#   sbatch run_opensocialjax.sh IPPO open_cleanup SEED=902 ENV_KWARGS.reveal_rule=true
# Everything after algo/env is forwarded to Hydra as key=value overrides.

ALGO="${1:-IPPO}"
ENV="${2:-open_cleanup}"
shift 2 2>/dev/null || shift $# 2>/dev/null

source "${CONDA_SH:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate "${CONDA_ENV:-OpenSocialJax}"

cd "$(dirname "$0")"
export PYTHONPATH=$PWD:$PYTHONPATH
ulimit -c 0

python algorithms/train.py --algo "$ALGO" --env "$ENV" "$@"
