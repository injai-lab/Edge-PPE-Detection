# Edge PPE Detection

Raspberry Pi 5와 Coral Edge TPU 환경에서 안전모, 안전조끼, 작업자 착용 상태를 감지하는 PPE detection 실행 패키지입니다. 2026-04-12 기준으로 정리한 EfficientDet Lite 기반 안정 버전과 MoveNet pose overlay 옵션을 포함합니다.

## 구성

```text
.
├── test_ppe_rpi5.py
├── run_original_ppe_camera.sh
├── run_original_ppe_camera_fast.sh
├── run_original_ppe_camera_movenet_headless.sh
├── run_original_ppe_camera_movenet_paper.sh
├── models/
│   └── effdet_lite0_ppe_stage2_ppe3_raw_int8_edgetpu.tflite
├── pose_models/
│   └── movenet_lightning_int8.tflite
└── docs/
```

## 실행 환경

- Raspberry Pi 5
- Python 3.9
- Coral Edge TPU runtime / `pycoral`
- `tflite-runtime`
- `numpy`, `Pillow`
- USB camera 또는 Picamera2 compatible CSI camera
- Preview 사용 시 OpenCV Python binding

기본 실행 스크립트는 기존 장비 기준 경로를 사용합니다.

- Python: `/home/team03/python39/bin/python3.9`
- Coral Python path: `/home/team03/coral_py39/usr/lib/python3/dist-packages`

다른 경로를 쓰는 경우 환경변수로 덮어쓸 수 있습니다.

```bash
PYTHON_BIN=/usr/bin/python3 \
PYTHONPATH_DIR=/usr/lib/python3/dist-packages \
./run_original_ppe_camera.sh
```

## 빠른 실행

```bash
chmod +x *.sh

# 기본 미리보기
./run_original_ppe_camera.sh

# 더 가벼운 미리보기
./run_original_ppe_camera_fast.sh

# 논문용 표시 스타일 + MoveNet
./run_original_ppe_camera_movenet_paper.sh

# 헤드리스 확인
./run_original_ppe_camera_movenet_headless.sh --headless
```

## 주요 옵션

스크립트 인자는 `test_ppe_rpi5.py`로 그대로 전달됩니다.

```bash
./run_original_ppe_camera.sh --camera-index 1
./run_original_ppe_camera.sh --person-threshold 0.42
./run_original_ppe_camera_movenet_paper.sh --pose-every 3
```

자주 조정하는 값은 환경변수로도 바꿀 수 있습니다.

```bash
EDGETPU_DEVICE=usb:0 PREVIEW_EVERY=3 PREVIEW_MAX_FPS=8 ./run_original_ppe_camera.sh
```

## 문서

- `docs/BENCHMARK_SUMMARY_CURRENT_20260412.md`: 현재 벤치마크 요약
- `docs/논문용_벤치마크_정리_20260412.md`: 논문용 벤치마크 정리
- `docs/논문용_재측정_메모_20260412.md`: 재측정 메모
- `docs/ORIGINAL_PACKAGE_README.md`: 백업 압축에 포함되어 있던 원본 README

## Contributor
- InJae_AI
