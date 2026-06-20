#!/bin/bash
# NO-HUMAN ablation: evaluate on the same imputed scenes + LLM language, but with
# NO human mesh injected. Measures how much accuracy depends on the gesture/human
# cue vs. language + 3D scene alone.
#
# Usage:
#   ./scripts/run_nohuman_eval.sh                 # single no-human eval (val)
#   ANGLE=1 ./scripts/run_nohuman_eval.sh         # also bucket IoU by arm angle
#   SPLIT=test ./scripts/run_nohuman_eval.sh
#   CKPT=/path/to/model.ckpt ./scripts/run_nohuman_eval.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate ges3vig-gpu

SPLIT=${SPLIT:-val}
SEED=${SEED:-123}
GPU=${GPU:-0}
CKPT=${CKPT:-/nas_hcfc/HRI_general/common/checkpoints/Ges3ViG_ImputeRefer.ckpt}
EXP=${EXP:-nohuman_eval_${SPLIT}}

cd "$(dirname "$0")/.."

EXTRA=()
if [ "${ANGLE:-0}" = "1" ]; then
    EXTRA+=(data.evaluator._target_=ges3vig.evaluation.scanrefer_angle_evaluator.ScanReferAngleEvaluator)
fi

CUDA_VISIBLE_DEVICES=${GPU} python test.py \
    data=nohumanimputerefer \
    experiment_name="${EXP}" \
    ckpt_path="${CKPT}" \
    data.inference.split="${SPLIT}" \
    test_seed="${SEED}" \
    "${EXTRA[@]}"
