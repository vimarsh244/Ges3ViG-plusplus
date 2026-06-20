#!/bin/bash
# Standard validation evaluator for the ORIGINAL (no-elbow / straight-arm) ImputeRefer
# dataset, reporting IoU@0.25 / IoU@0.50 (overall + unique/multiple).
#
# Reproducible by default (fixed seed). To run with a random seed, pass SEED=random.
#
# Usage:
#   ./scripts/run_val_eval.sh                       # noelbowimputerefer, val, fixed seed=123
#   DATA=imputerefer ./scripts/run_val_eval.sh      # bent-elbow dataset instead
#   SPLIT=test ./scripts/run_val_eval.sh
#   SEED=random ./scripts/run_val_eval.sh           # non-deterministic seed
#   CKPT=/path/to/model.ckpt ./scripts/run_val_eval.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate ges3vig-gpu

DATA=${DATA:-noelbowimputerefer}
SPLIT=${SPLIT:-val}
SEED=${SEED:-123}
GPU=${GPU:-0}
CKPT=${CKPT:-/nas_hcfc/HRI_general/common/checkpoints/Ges3ViG_ImputeRefer.ckpt}
EXP=${EXP:-val_eval_${DATA}_${SPLIT}}

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=${GPU} python test.py \
    data="${DATA}" \
    experiment_name="${EXP}" \
    ckpt_path="${CKPT}" \
    data.inference.split="${SPLIT}" \
    test_seed="${SEED}"
