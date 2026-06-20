#!/bin/bash
# Sensitivity scan across the elevation / elbow ARM ANGLE range.
# Runs a single inference pass and breaks down IoU@0.25 / IoU@0.50 by the arm
# angle used when composing each sample (default bins: 20-50, 50-80, 80-110,
# 110-150 deg, plus a per-10-deg table).
#
# NOTE: arm-angle variation only exists in the bent-elbow `imputerefer` dataset.
# The no-elbow dataset uses a single straight-arm pose, so this scan is most
# meaningful with DATA=imputerefer (the default below).
#
# Usage:
#   ./scripts/run_angle_sensitivity.sh
#   SPLIT=test ./scripts/run_angle_sensitivity.sh
#   CKPT=/path/to/model.ckpt ./scripts/run_angle_sensitivity.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate ges3vig-gpu

DATA=${DATA:-imputerefer}
SPLIT=${SPLIT:-val}
SEED=${SEED:-123}
GPU=${GPU:-0}
CKPT=${CKPT:-/nas_hcfc/HRI_general/common/checkpoints/Ges3ViG_ImputeRefer.ckpt}
EXP=${EXP:-angle_sensitivity_${DATA}_${SPLIT}}

cd "$(dirname "$0")/.."

CUDA_VISIBLE_DEVICES=${GPU} python test.py \
    data="${DATA}" \
    experiment_name="${EXP}" \
    ckpt_path="${CKPT}" \
    data.inference.split="${SPLIT}" \
    test_seed="${SEED}" \
    data.evaluator._target_=ges3vig.evaluation.scanrefer_angle_evaluator.ScanReferAngleEvaluator
