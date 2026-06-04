#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/team03/python39/bin/python3.9}"
PYTHONPATH_DIR="${PYTHONPATH_DIR:-/home/team03/coral_py39/usr/lib/python3/dist-packages}"
MODEL_PATH="${SCRIPT_DIR}/models/effdet_lite0_ppe_stage2_ppe3_raw_int8_edgetpu.tflite"
POSE_MODEL_PATH="${SCRIPT_DIR}/pose_models/movenet_lightning_int8.tflite"
SCRIPT_PATH="${SCRIPT_DIR}/test_ppe_rpi5.py"
EDGETPU_DEVICE="${EDGETPU_DEVICE:-auto-pci}"
POSE_EVERY="${POSE_EVERY:-2}"
PERSON_THRESHOLD="${PERSON_THRESHOLD:-0.38}"
POSE_PERSON_THRESHOLD="${POSE_PERSON_THRESHOLD:-0.35}"

if [[ -n "${PYTHONPATH:-}" ]]; then
  export PYTHONPATH="${PYTHONPATH_DIR}:${PYTHONPATH}"
else
  export PYTHONPATH="${PYTHONPATH_DIR}"
fi

exec "${PYTHON_BIN}" -u "${SCRIPT_PATH}" \
  --camera \
  --camera-backend opencv \
  --camera-index 0 \
  --camera-width 640 \
  --camera-height 480 \
  --camera-fps 30 \
  --camera-buffer-size 1 \
  --camera-fourcc MJPG \
  --camera-read-mode latest \
  --delegate edgetpu \
  --edgetpu-device "${EDGETPU_DEVICE}" \
  --model "${MODEL_PATH}" \
  --label-preset default \
  --detector-resize-mode letterbox \
  --person-threshold "${PERSON_THRESHOLD}" \
  --show-compliant \
  --paper-style \
  --pose-model "${POSE_MODEL_PATH}" \
  --pose-delegate cpu \
  --pose-person-threshold "${POSE_PERSON_THRESHOLD}" \
  --pose-every "${POSE_EVERY}" \
  --pose-all-persons \
  "$@"
