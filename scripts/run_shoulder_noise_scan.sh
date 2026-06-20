#!/bin/bash
# Sensitivity scan over SHOULDER-ROTATION NOISE applied to the imputed human.
#
# The human is randomly rotated about its shoulder by +/- (DEG/2). Internally the
# config takes a "percentage" P where  DEG = P * 1.8  (i.e. P = DEG / 1.8).
# The default operating point of the dataset is P=5 -> 9 degrees.
#
# This sweeps a list of degree values and runs a full IoU evaluation at each,
# so you can see how accuracy degrades as gesture/pose noise increases.
#
# Usage:
#   ./scripts/run_shoulder_noise_scan.sh
#   DATA=imputerefer ./scripts/run_shoulder_noise_scan.sh
#   DEGREES="0 9 18 27 36" ./scripts/run_shoulder_noise_scan.sh
#   CKPT=/path/to/model.ckpt ./scripts/run_shoulder_noise_scan.sh
set -e
source ~/miniconda3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh
conda activate ges3vig-gpu

DATA=${DATA:-noelbowimputerefer}
SPLIT=${SPLIT:-val}
SEED=${SEED:-123}
GPU=${GPU:-0}
CKPT=${CKPT:-/nas_hcfc/HRI_general/common/checkpoints/Ges3ViG_ImputeRefer.ckpt}
# Default sweep includes the dataset default of 9 degrees.
DEGREES=${DEGREES:-"0 6 9 12 18 24 30 36 42 48"}

cd "$(dirname "$0")/.."

for deg in ${DEGREES}; do
    perc=$(awk "BEGIN {printf \"%.4f\", $deg / 1.8}")
    echo "================================================================"
    echo "Shoulder-rotation noise = ${deg} deg  (human_perturbation_percentage = ${perc})"
    echo "================================================================"
    CUDA_VISIBLE_DEVICES=${GPU} python test.py \
        data="${DATA}" \
        experiment_name="shoulder_noise_${DATA}_deg${deg}" \
        ckpt_path="${CKPT}" \
        data.inference.split="${SPLIT}" \
        test_seed="${SEED}" \
        data.human_perturbation_percentage="${perc}"
done
