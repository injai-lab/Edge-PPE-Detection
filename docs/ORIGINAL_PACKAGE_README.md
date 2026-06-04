# final_Efficient

2026-04-12 기준으로 정리한 원래 Efficient 기반 안정 버전 패키지입니다.

포함 파일
- `test_ppe_rpi5.py`
- `models/effdet_lite0_ppe_stage2_ppe3_raw_int8_edgetpu.tflite`
- `pose_models/movenet_lightning_int8.tflite`
- `run_original_ppe_camera.sh`
- `run_original_ppe_camera_fast.sh`
- `run_original_ppe_camera_movenet_headless.sh`
- `run_original_ppe_camera_movenet_paper.sh`

추천 실행
- 기본 미리보기: `./run_original_ppe_camera.sh`
- 더 가벼운 미리보기: `./run_original_ppe_camera_fast.sh`
- 논문 느낌 표시 + MoveNet: `./run_original_ppe_camera_movenet_paper.sh`
- 헤드리스 확인: `./run_original_ppe_camera_movenet_headless.sh --headless`

Pi 배치 예시
```bash
scp -r final_Efficient team03@192.168.137.94:~/newppe_probe/
ssh team03@192.168.137.94
cd ~/newppe_probe/final_Efficient
chmod +x *.sh
./run_original_ppe_camera.sh
```

주의
- 실행은 Raspberry Pi 쪽 Python 3.9 + Coral pycoral 환경을 전제로 합니다.
- shell 스크립트는 패키지 내부 상대 경로를 쓰도록 정리되어 있습니다.
