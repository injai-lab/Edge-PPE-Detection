# EdgePPE-Net Benchmark Summary (2026-04-12)

## 1. Detection AP@0.5

- Test set: `coco_test/_annotations.coco.json`
- Images: 34
- Classes: `worker`, `helmet`, `vest`
- Confidence thresholds used in the current script:
  - `worker/person = 0.38`
  - `helmet = 0.28`
  - `vest = 0.28`

For a fair hardware comparison, the same EdgeTPU-compiled detector was used on both Coral USB and PCIe HAT. The CPU row uses the canonical CPU INT8 model.

| Runtime | Model | worker AP@0.5 | helmet AP@0.5 | vest AP@0.5 | mAP@0.5 | Missed GT | Mean infer |
|---|---|---:|---:|---:|---:|---:|---:|
| CPU | `/home/team03/rpi5_deploy/effdet_lite0_ppe_stage2_ppe3.tflite` | 0.1662 | 0.7534 | 0.4026 | 0.4408 | 1 | 26.38 ms/image |
| Coral USB | `/home/team03/rpi5_deploy/edgetpu_build/effdet_lite0_ppe_stage2_ppe3_edgetpu.tflite` on `usb:0` | 0.1672 | 0.7576 | 0.4026 | 0.4424 | 1 | 62.62 ms/image |
| PCIe HAT | `/home/team03/rpi5_deploy/edgetpu_build/effdet_lite0_ppe_stage2_ppe3_edgetpu.tflite` on `pci:0` | 0.1672 | 0.7576 | 0.4026 | 0.4424 | 1 | 21.45 ms/image |

Notes:
- The same compiled EdgeTPU model preserved the CPU detector accuracy on both USB and PCIe.
- On this Pi setup, PCIe HAT was much faster than Coral USB for the same detector.

## 2. Table 3: Box IoU Only vs EdgePPE-Net

- Pipeline used here: current tuned `final_Efficient` deployment script
- Detector: `/home/team03/newppe_probe/final_Efficient/models/effdet_lite0_ppe_stage2_ppe3_raw_int8_edgetpu.tflite`
- Pose model: `/home/team03/newppe_probe/final_Efficient/pose_models/movenet_lightning_int8.tflite`
- Detector delegate: PCIe EdgeTPU
- Pose delegate: CPU, 2 threads
- Pose mode: conditional pose (`pose_every = 2`)

### Current GT subset

- Frames: 20
- Worker instances: 58
- Included scenarios:
  - `normal_wear`: 20
  - `adjacent_overlap`: 17
  - `missing_helmet_only`: 8
  - `missing_vest_only`: 13
- Still missing:
  - hand-held PPE case
  - both-missing case

### Overall metrics

| Method | Precision | Recall | F1 | FP Rate |
|---|---:|---:|---:|---:|
| Box IoU only | 0.8500 | 0.8095 | 0.8293 | 0.0811 |
| EdgePPE-Net | 1.0000 | 0.6667 | 0.8000 | 0.0000 |

Interpretation:
- The current tuned EdgePPE-Net configuration removed all false positives on this subset.
- The gain came with a recall drop, so F1 was slightly lower than the box-only baseline.
- On the `adjacent_overlap` subset, false positives changed from 3 to 0.

## 3. Conditional Pose Execution Savings

- Videos: 4 staged videos
- Frames processed: 768
- Detector: current tuned PCIe deployment pipeline
- Pose: MoveNet Lightning INT8 on CPU, 2 threads

### Aggregate

| Metric | Value |
|---|---:|
| Mean detected workers/frame | 2.7930 |
| Mean pose executions/frame | 0.5990 |
| Pose skip ratio | 78.55% |
| Mean MoveNet latency | 20.02 ms/run |
| Estimated saved pose time | 43.92 ms/frame |
| Actual infer-only FPS estimate | 42.75 |
| Exhaustive pose infer-only FPS estimate | 14.86 |

### Totals

| Metric | Value |
|---|---:|
| Detected workers total | 2145 |
| Pose candidates total | 914 |
| Pose executions total | 460 |
| Pose elapsed total | 9208.80 ms |
| Detector infer total | 8757.57 ms |
| Exhaustive pose time estimate | 42941.03 ms |
| Saved pose time estimate | 33732.23 ms |

### Per-video breakdown

| Video | Frames | Detected workers | Pose executions |
|---|---:|---:|---:|
| `site_workers_1.mp4` | 192 | 720 | 4 |
| `site_workers_2.mp4` | 192 | 618 | 67 |
| `no_helmet_site.mp4` | 192 | 315 | 148 |
| `no_vest_site.mp4` | 192 | 492 | 241 |

## 4. Recommendation for the Paper

- Use Section 4 detection accuracy table from the fair detector benchmark above:
  - CPU
  - Coral USB
  - PCIe HAT
- Use Table 3 and pose-savings numbers from the current tuned deployment pipeline.
- Add one sentence that the current Table 3 subset is still incomplete because it does not yet include:
  - hand-held PPE false-positive scenes
  - both-missing scenes

## 5. Key Output Files

- Detection metrics:
  - `current_run/detection_cpu_metrics_current.json`
  - `current_run/detection_usb_metrics_current.json`
  - `current_run/detection_pcie_same_edgetpu_model_metrics_current.json`
- Table 3 metrics:
  - `current_run/table3_box_metrics_current.json`
  - `current_run/table3_full_metrics_current.json`
- Pose savings raw output:
  - `current_run/pose_full_videos_pcie_current.json`
