"""PPE Detection test script for Raspberry Pi 5.

Supports:
  - image inference
  - CSI camera via Picamera2
  - USB webcams via OpenCV
  - optional Coral Edge TPU delegate
  - optional MoveNet pose overlay on detected persons
"""

import argparse
import glob
import math
import threading
import time
from ctypes.util import find_library
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

try:
    from tflite_runtime.interpreter import Interpreter, load_delegate
except ImportError:
    from tensorflow.lite import Interpreter

    def load_delegate(*args, **kwargs):  # type: ignore[no-redef]
        raise RuntimeError("Edge TPU delegate requires tflite-runtime.")

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = str(BASE_DIR / "models" / "effdet_lite0_ppe_stage2_ppe3_raw_int8_edgetpu.tflite")
DUAL_MODEL_SEG0_PATH = str(BASE_DIR / "int8_coral_TPU" / "effdet_lite0_ppe_stage2_ppe3_segment_0_of_2_edgetpu.tflite")
DUAL_MODEL_SEG1_PATH = str(BASE_DIR / "int8_coral_TPU" / "effdet_lite0_ppe_stage2_ppe3_segment_1_of_2_edgetpu.tflite")
DEFAULT_EDGETPU_DEVICE = "auto-pci"
DEFAULT_LABELS = {0: "helmet", 1: "vest", 2: "person"}
FINAL_FT_B8_LABELS = {0: "person", 1: "vest", 2: "helmet"}
LABEL_PRESETS = {
    "default": DEFAULT_LABELS,
    "final_ft_b8": FINAL_FT_B8_LABELS,
}
LABELS = dict(DEFAULT_LABELS)
COLORS = {"person": (0, 120, 255), "helmet": (0, 200, 0), "vest": (255, 165, 0)}
ALERT_COLOR = (255, 0, 0)
LETTERBOX_PAD_VALUE = 114
DUAL_OUTPUT_STRIDE = 4
CONF_THRESHOLD = 0.25
CONF_THRESHOLDS = {
    "person": 0.38,
    "helmet": 0.28,
    "vest": 0.28,
}
NMS_IOU_THRESHOLD = 0.5
MAX_DETECTIONS = 25
ANCHOR_SCALE = 4.0
ASPECT_RATIOS = (1.0, 2.0, 0.5)
NUM_SCALES = 3
BOX_SCALE = (10.0, 10.0, 5.0, 5.0)
RAW_DECODE_PROFILES = {
    "legacy": {
        "anchor_scale": ANCHOR_SCALE,
        "box_scale": BOX_SCALE,
        "exp_clip": None,
    },
    "raw_uint8_direct": {
        "anchor_scale": 3.0,
        "box_scale": (1.0, 1.0, 1.0, 1.0),
        "exp_clip": 4.0,
    },
}
MATCH_SCORE_MIN = 0.05
HELMET_REGION_TOP = -0.12
HELMET_REGION_BOTTOM = 0.30
VEST_REGION_TOP = 0.15
VEST_REGION_BOTTOM = 0.70
PREVIEW_MIN_SCALE = 0.5
PREVIEW_MAX_SCALE = 4.0
PREVIEW_SCALE_STEP = 0.25
PERSON_DISPLAY_SHRINK_X = 0.08
PERSON_DISPLAY_SHRINK_TOP = 0.02
PERSON_DISPLAY_SHRINK_BOTTOM = 0.01
PERSON_DISPLAY_POSE_PAD_X = 0.05
PERSON_DISPLAY_POSE_PAD_TOP = 0.05
PERSON_DISPLAY_POSE_PAD_BOTTOM = 0.06
PAPER_PERSON_SHRINK_X = 0.08
PAPER_PERSON_SHRINK_TOP = 0.01
PAPER_PERSON_SHRINK_BOTTOM = 0.00
PAPER_VIOLATION_PERSON_SHRINK_X = 0.14
PAPER_VIOLATION_PERSON_SHRINK_TOP = 0.02
PAPER_VIOLATION_PERSON_SHRINK_BOTTOM = 0.02
PAPER_COMPLIANT_PERSON_SHRINK_X = 0.16
PAPER_COMPLIANT_PERSON_SHRINK_TOP = 0.02
PAPER_COMPLIANT_PERSON_SHRINK_BOTTOM = 0.02
PAPER_PERSON_BOX_PAD_X = 0.01
PAPER_PERSON_BOX_PAD_TOP = 0.02
PAPER_PERSON_BOX_PAD_BOTTOM = 0.03
PAPER_VIOLATION_PERSON_BOX_PAD_X = 0.006
PAPER_VIOLATION_PERSON_BOX_PAD_TOP = 0.015
PAPER_VIOLATION_PERSON_BOX_PAD_BOTTOM = 0.02
PAPER_PERSON_POSE_PAD_X = 0.06
PAPER_PERSON_POSE_PAD_TOP = 0.08
PAPER_PERSON_POSE_PAD_BOTTOM = 0.10
HELMET_MATCH_X_MARGIN = 0.14
VEST_MATCH_X_MARGIN = 0.10
HELMET_MATCH_SCORE_MIN = 0.04
VEST_MATCH_SCORE_MIN = 0.04
PERSON_DUPLICATE_IOU_THRESHOLD = 0.48
PERSON_DUPLICATE_CONTAINMENT_THRESHOLD = 0.72
PARTIAL_PERSON_EDGE_MARGIN = 0.02
PARTIAL_PERSON_MIN_AREA = 0.30
PARTIAL_PERSON_MIN_WIDTH = 0.45
PARTIAL_PERSON_MIN_HEIGHT = 0.45
PARTIAL_PERSON_STRICT_AREA = 0.18
PARTIAL_PERSON_STRICT_WIDTH = 0.30
PARTIAL_PERSON_STRICT_HEIGHT = 0.55
PARTIAL_PERSON_MIN_ASPECT_RATIO = 1.05
PARTIAL_PERSON_POSE_SCORE_THRESHOLD = 0.35
PERSON_SHAPE_MIN_ASPECT_RATIO = 1.15
PERSON_SHAPE_MIN_HEIGHT = 0.18
PERSON_SHAPE_MIN_AREA = 0.022
PERSON_SHAPE_MAX_WIDTH = 0.42
PERSON_SHAPE_WIDE_MIN_ASPECT_RATIO = 1.35
PERSON_SHAPE_POSE_MIN_SCORE = 0.50
PERSON_SHAPE_NO_POSE_MIN_SCORE = 0.60
PERSON_SHAPE_STRONG_POSE_SCORE_THRESHOLD = 0.45
PERSON_SHAPE_MIN_VISIBLE_KEYPOINTS = 6
POSE_CACHE_MIN_IOU = 0.45
STABLE_PERSON_MIN_IOU = 0.25
STABLE_PERSON_HOLD_FRAMES = 2
STABLE_PERSON_MIN_HITS = 1
STABLE_PERSON_MIN_SCORE = 0.42
STABLE_PERSON_DECAY = 0.96
STABLE_PERSON_MIN_AREA = 0.08
POSE_KEYPOINT_NAMES = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]
POSE_EDGES = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (0, 5),
    (0, 6),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
POSE_EDGE_COLORS = [
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (0, 255, 255),
    (255, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (0, 255, 255),
    (255, 255, 0),
    (255, 0, 255),
    (255, 0, 255),
    (0, 255, 255),
    (0, 255, 255),
]
POSE_HEAD_KEYPOINT_IDS = (0, 1, 2, 3, 4)
POSE_SHOULDER_KEYPOINT_IDS = (5, 6)
POSE_HIP_KEYPOINT_IDS = (11, 12)
POSE_TORSO_KEYPOINT_IDS = (5, 6, 11, 12)


def normalize_edgetpu_device(device):
    if device is None:
        return None
    value = str(device).strip().lower()
    if not value or value in {"auto", "default", "none"}:
        return None
    return value


def get_edgetpu_candidate_devices(device):
    normalized = normalize_edgetpu_device(device)
    if normalized == "auto-pci":
        return ["pci:0", "pci:1", "pci:2", "pci:3"]
    if normalized == "auto-usb":
        return ["usb:0", "usb:1", "usb:2", "usb:3"]
    if normalized is None:
        return [None]
    return [normalized]


def format_edgetpu_device(device):
    return device if device else "default"


def describe_local_edgetpu_state():
    apex_devices = sorted(glob.glob("/dev/apex_*"))
    details = [f"apex nodes: {', '.join(apex_devices) if apex_devices else 'none'}"]
    return "; ".join(details)


def build_interpreter(model_path, num_threads=4, delegate="cpu", edgetpu_device=None):
    delegate = delegate.lower()

    if delegate == "edgetpu":
        lib_name = find_library("edgetpu") or "libedgetpu.so.1"
        candidates = get_edgetpu_candidate_devices(edgetpu_device)
        failures = []

        for candidate in candidates:
            kwargs = {}
            delegate_options = {"device": candidate} if candidate else {}
            try:
                kwargs["experimental_delegates"] = [load_delegate(lib_name, delegate_options)]
                if candidate:
                    print(f"Using delegate: Edge TPU ({lib_name}, device={candidate})")
                else:
                    print(f"Using delegate: Edge TPU ({lib_name})")
                if "edgetpu" not in Path(model_path).stem.lower():
                    print("WARNING: model filename does not look Edge TPU-compiled; CPU may be faster.")
                return Interpreter(model_path=model_path, **kwargs)
            except Exception as exc:
                failures.append(f"{format_edgetpu_device(candidate)} -> {exc}")

        tried = ", ".join(format_edgetpu_device(candidate) for candidate in candidates)
        local_state = describe_local_edgetpu_state()
        raise RuntimeError(
            "Failed to initialize Edge TPU delegate. "
            f"Tried devices: {tried}. "
            f"Local state: {local_state}. "
            f"Errors: {' | '.join(failures)}. "
            "If you are using a dual PCIe Coral, try --edgetpu-device auto-pci or --edgetpu-device pci:1."
        )
    else:
        kwargs = {"num_threads": num_threads}
        print(f"Using delegate: CPU ({num_threads} threads)")

    return Interpreter(model_path=model_path, **kwargs)


def detect_raw_model_layout(output_details, num_classes):
    if len(output_details) == 2 and output_details[0]["shape"][-1] == num_classes:
        return "concat"

    class_channels = 9 * num_classes
    box_channels = 9 * 4
    class_count = 0
    box_count = 0
    for details in output_details:
        shape = tuple(int(v) for v in details["shape"])
        if len(shape) != 4:
            continue
        if shape[-1] == class_channels:
            class_count += 1
        elif shape[-1] == box_channels:
            box_count += 1

    if class_count >= 3 and class_count == box_count:
        return "multiscale"

    return None


def load_model(model_path, num_threads=4, delegate="cpu", edgetpu_device=None):
    interp = build_interpreter(
        model_path,
        num_threads=num_threads,
        delegate=delegate,
        edgetpu_device=edgetpu_device,
    )
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    h, w = input_details[0]["shape"][1], input_details[0]["shape"][2]
    raw_layout = detect_raw_model_layout(output_details, len(LABELS))
    is_raw_model = raw_layout is not None
    model_info = {"is_raw_model": is_raw_model}
    if is_raw_model:
        raw_profile_name = resolve_raw_decode_profile(model_path)
        raw_profile = RAW_DECODE_PROFILES[raw_profile_name]
        model_info["raw_layout"] = raw_layout
        model_info["raw_decode_profile"] = raw_profile_name
        model_info["raw_profile"] = raw_profile
        model_info["anchors"] = generate_anchors(h, w, anchor_scale=raw_profile["anchor_scale"])
    print(f"Model loaded: {model_path}")
    print(f"  Input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"  Threads: {num_threads}")
    if is_raw_model:
        output_kind = f"raw:{raw_layout}"
    else:
        output_kind = "postprocessed"
    print(f"  Output tensors: {len(output_details)} ({output_kind})")
    if is_raw_model:
        print(
            "  Raw decode profile: "
            f"{model_info['raw_decode_profile']} "
            f"(anchor_scale={model_info['raw_profile']['anchor_scale']}, "
            f"box_scale={model_info['raw_profile']['box_scale']})"
        )
    return interp, input_details, output_details, (h, w), model_info


def load_dual_model(seg0_path, seg1_path, device0="pci:0", device1="usb:0"):
    try:
        from pycoral.pipeline.pipelined_model_runner import PipelinedModelRunner
        from pycoral.utils.edgetpu import list_edge_tpus, make_interpreter
    except ImportError as exc:
        raise RuntimeError(
            "Dual Edge TPU mode requires PyCoral. Use the Python 3.9 wrapper environment or run via run_ppe_dual_tpu.sh."
        ) from exc

    interp0 = make_interpreter(seg0_path, device=device0)
    interp1 = make_interpreter(seg1_path, device=device1)
    interp0.allocate_tensors()
    interp1.allocate_tensors()

    input_details = interp0.get_input_details()
    output_details = interp1.get_output_details()
    h, w = input_details[0]["shape"][1], input_details[0]["shape"][2]
    model_info = {
        "is_raw_model": False,
        "runtime": "edgetpu_dual",
        "dual_output_stride": DUAL_OUTPUT_STRIDE,
    }
    runner = {
        "runner": PipelinedModelRunner([interp0, interp1]),
        "input_name": input_details[0]["name"],
        "device0": device0,
        "device1": device1,
        "segments": (seg0_path, seg1_path),
    }
    print("Using delegate: Dual Edge TPU pipeline")
    print(f"  Segment 0: {seg0_path} on {device0}")
    print(f"  Segment 1: {seg1_path} on {device1}")
    print(f"  Edge TPUs: {list_edge_tpus()}")
    print(f"  Input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"  Output tensors: {len(output_details)} (postprocessed, stride={DUAL_OUTPUT_STRIDE})")
    return runner, input_details, output_details, (h, w), model_info


def load_runtime(args):
    if args.delegate == "edgetpu_dual":
        seg0_path = resolve_input_path(args.dual_model_seg0)
        seg1_path = resolve_input_path(args.dual_model_seg1)
        runtime = load_dual_model(seg0_path, seg1_path, device0=args.dual_device0, device1=args.dual_device1)
    else:
        runtime = load_model(args.model, args.threads, args.delegate, edgetpu_device=args.edgetpu_device)

    runtime[4]["resize_mode"] = args.detector_resize_mode
    return runtime


def load_optional_person_runtime(args):
    if not getattr(args, "person_model", None):
        return None
    if args.delegate == "edgetpu_dual":
        raise RuntimeError("Hybrid person model is not supported with --delegate edgetpu_dual.")

    person_model_path = resolve_input_path(args.person_model)
    preset_name, labels_map = get_label_map_for_preset(args.person_model_label_preset, person_model_path)
    runtime = load_model(
        person_model_path,
        args.threads,
        args.delegate,
        edgetpu_device=args.edgetpu_device,
    )
    runtime[4]["resize_mode"] = args.person_model_resize_mode
    print(
        "Hybrid person source: "
        f"{person_model_path} "
        f"(preset={preset_name}, threshold={float(args.person_model_threshold):.2f}, "
        f"resize={args.person_model_resize_mode})"
    )
    return {
        "runtime": runtime,
        "labels_map": labels_map,
        "threshold": float(args.person_model_threshold),
    }


def close_runtime(interp, model_info):
    if model_info.get("runtime") == "edgetpu_dual":
        try:
            interp["runner"].push({})
            interp["runner"]._runner.Pop()
        except Exception:
            pass
        try:
            interp["runner"]._runner = None
        except Exception:
            pass


def load_pose_model(model_path, num_threads=2, delegate="cpu", edgetpu_device=None):
    interp = build_interpreter(
        model_path,
        num_threads=num_threads,
        delegate=delegate,
        edgetpu_device=edgetpu_device,
    )
    interp.allocate_tensors()
    input_details = interp.get_input_details()
    output_details = interp.get_output_details()
    h, w = input_details[0]["shape"][1], input_details[0]["shape"][2]
    print(f"Pose model loaded: {model_path}")
    print(f"  Pose input: {input_details[0]['shape']} ({input_details[0]['dtype']})")
    print(f"  Pose output tensors: {len(output_details)}")
    return {
        "interp": interp,
        "input_details": input_details,
        "output_details": output_details,
        "input_size": (h, w),
    }


def resolve_input_path(path_value):
    if not path_value:
        return path_value

    candidate = Path(path_value).expanduser()
    if candidate.is_absolute():
        return str(candidate)

    script_relative = BASE_DIR / candidate
    if script_relative.exists():
        return str(script_relative)

    return str(candidate)


def resolve_output_video_path(input_video, output_value=None):
    if output_value:
        out_path = Path(output_value).expanduser()
        if out_path.is_absolute():
            return str(out_path)
        return str(Path.cwd() / out_path)

    video_path = Path(input_video).expanduser()
    stem = video_path.stem + "_result.mp4"
    return str(video_path.with_name(stem))


def configure_label_preset(label_preset, model_hint=""):
    global LABELS

    selected = resolve_label_preset_name(label_preset, model_hint)

    LABELS = dict(LABEL_PRESETS[selected])
    print(f"Label preset: {selected} -> {LABELS}")
    return selected


def resolve_label_preset_name(label_preset, model_hint=""):
    selected = label_preset
    if selected == "auto":
        stem = Path(model_hint).stem.lower() if model_hint else ""
        selected = "final_ft_b8" if "final_ft_b8" in stem else "default"
    return selected


def get_label_map_for_preset(label_preset, model_hint=""):
    selected = resolve_label_preset_name(label_preset, model_hint)
    return selected, dict(LABEL_PRESETS[selected])


def apply_conf_threshold_overrides(args):
    overrides = {}
    if getattr(args, "person_threshold", None) is not None:
        overrides["person"] = float(args.person_threshold)
    if getattr(args, "helmet_threshold", None) is not None:
        overrides["helmet"] = float(args.helmet_threshold)
    if getattr(args, "vest_threshold", None) is not None:
        overrides["vest"] = float(args.vest_threshold)
    if not overrides:
        return
    CONF_THRESHOLDS.update(overrides)
    print(f"Threshold overrides: {CONF_THRESHOLDS}")


def collect_detection_records(boxes, classes, scores, labels_map=None, include_labels=None, thresholds=None):
    labels_map = labels_map or LABELS
    thresholds = thresholds or CONF_THRESHOLDS
    include_labels = set(include_labels) if include_labels is not None else None

    records = []
    for i in range(len(scores)):
        cls_id = int(classes[i])
        label = labels_map.get(cls_id, f"cls{cls_id}")
        if include_labels is not None and label not in include_labels:
            continue
        threshold = thresholds.get(label, CONF_THRESHOLD)
        if float(scores[i]) < threshold:
            continue
        records.append(
            {
                "class_id": cls_id,
                "label": label,
                "box": np.array(boxes[i], dtype=np.float32),
                "score": float(scores[i]),
            }
        )
    return records


def build_person_views_from_records(records):
    persons = filter_person_records([r for r in records if r["label"] == "person"])
    helmets = [r for r in records if r["label"] == "helmet"]
    vests = [r for r in records if r["label"] == "vest"]
    base_person_views = [{"box": person["box"], "score": person["score"]} for person in persons]
    return apply_ppe_matches(base_person_views, helmets, vests)


def detection_records_to_arrays(records, target_labels=None):
    target_labels = target_labels or LABELS
    inverse_labels = {label: cls_id for cls_id, label in target_labels.items()}
    filtered = [record for record in records if record["label"] in inverse_labels]
    if not filtered:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    boxes = np.array([record["box"] for record in filtered], dtype=np.float32)
    classes = np.array([inverse_labels[record["label"]] for record in filtered], dtype=np.int32)
    scores = np.array([record["score"] for record in filtered], dtype=np.float32)
    return boxes, classes, scores


def build_person_views_hybrid(
    boxes,
    classes,
    scores,
    person_boxes=None,
    person_classes=None,
    person_scores=None,
    person_labels_map=None,
    person_threshold=None,
):
    item_records = collect_detection_records(
        boxes,
        classes,
        scores,
        labels_map=LABELS,
        include_labels={"helmet", "vest"},
    )
    if person_boxes is None or person_classes is None or person_scores is None:
        person_records = collect_detection_records(
            boxes,
            classes,
            scores,
            labels_map=LABELS,
            include_labels={"person"},
        )
    else:
        person_thresholds = {"person": CONF_THRESHOLDS["person"] if person_threshold is None else float(person_threshold)}
        person_records = collect_detection_records(
            person_boxes,
            person_classes,
            person_scores,
            labels_map=person_labels_map or LABELS,
            include_labels={"person"},
            thresholds=person_thresholds,
        )

    merged_records = person_records + item_records
    merged_boxes, merged_classes, merged_scores = detection_records_to_arrays(merged_records, target_labels=LABELS)
    person_views = build_person_views_from_records(merged_records)
    return merged_boxes, merged_classes, merged_scores, person_views


def ensure_pil_image(image_or_frame):
    if isinstance(image_or_frame, Image.Image):
        return image_or_frame
    return Image.fromarray(np.asarray(image_or_frame, dtype=np.uint8))


def get_rgb_frame_array(image_or_frame):
    if isinstance(image_or_frame, Image.Image):
        return np.asarray(image_or_frame.convert("RGB"), dtype=np.uint8)
    return np.asarray(image_or_frame, dtype=np.uint8)


def resize_rgb_frame(frame_rgb, input_size):
    input_h, input_w = input_size
    if frame_rgb.shape[0] == input_h and frame_rgb.shape[1] == input_w:
        return np.ascontiguousarray(frame_rgb)

    try:
        import cv2

        resized = cv2.resize(frame_rgb, (input_w, input_h), interpolation=cv2.INTER_LINEAR)
    except Exception:
        resized = np.array(
            Image.fromarray(frame_rgb).resize((input_w, input_h), Image.BILINEAR),
            dtype=np.uint8,
        )
    return np.ascontiguousarray(resized)


def letterbox_rgb_frame(frame_rgb, input_size, pad_value=LETTERBOX_PAD_VALUE):
    input_h, input_w = input_size
    orig_h, orig_w = frame_rgb.shape[:2]
    if orig_h <= 0 or orig_w <= 0:
        raise RuntimeError(f"Invalid frame shape for letterbox: {frame_rgb.shape}")

    scale = min(input_w / float(orig_w), input_h / float(orig_h))
    resized_w = max(1, int(round(orig_w * scale)))
    resized_h = max(1, int(round(orig_h * scale)))
    resized = resize_rgb_frame(frame_rgb, (resized_h, resized_w))

    pad_left = int((input_w - resized_w) // 2)
    pad_top = int((input_h - resized_h) // 2)

    canvas = np.full((input_h, input_w, 3), int(pad_value), dtype=np.uint8)
    canvas[pad_top : pad_top + resized_h, pad_left : pad_left + resized_w] = resized
    transform = {
        "mode": "letterbox",
        "orig_h": orig_h,
        "orig_w": orig_w,
        "input_h": input_h,
        "input_w": input_w,
        "pad_top": float(pad_top),
        "pad_left": float(pad_left),
        "resized_h": float(resized_h),
        "resized_w": float(resized_w),
        "scale_y": resized_h / float(orig_h),
        "scale_x": resized_w / float(orig_w),
    }
    return canvas, transform


def prepare_detector_input(image_or_frame, input_details, input_size, resize_mode="letterbox"):
    frame_rgb = get_rgb_frame_array(image_or_frame)
    orig_h, orig_w = frame_rgb.shape[:2]

    if resize_mode == "letterbox":
        resized, transform = letterbox_rgb_frame(frame_rgb, input_size)
    else:
        resized = resize_rgb_frame(frame_rgb, input_size)
        transform = {
            "mode": "stretch",
            "orig_h": orig_h,
            "orig_w": orig_w,
            "input_h": int(input_size[0]),
            "input_w": int(input_size[1]),
        }

    dtype = input_details[0]["dtype"]

    if dtype == np.float32:
        prepared = resized.astype(np.float32) / 255.0
    elif dtype == np.uint8:
        prepared = resized.astype(np.uint8, copy=False)
    elif dtype == np.int8:
        scale, zero_point = input_details[0].get("quantization", (0.0, 0))
        if scale:
            prepared = np.clip(
                np.round((resized.astype(np.float32) / 255.0) / scale + zero_point),
                -128,
                127,
            ).astype(np.int8)
        else:
            prepared = np.clip(resized.astype(np.int16) - 128, -128, 127).astype(np.int8)
    else:
        raise RuntimeError(f"Unsupported detector input dtype: {dtype}")

    return np.expand_dims(np.ascontiguousarray(prepared), axis=0), transform


def remap_boxes_to_original(boxes, transform):
    boxes = np.asarray(boxes, dtype=np.float32)
    if boxes.size == 0:
        return boxes.reshape((-1, 4))

    boxes = boxes.reshape((-1, 4))
    mode = transform.get("mode", "stretch")
    if mode == "stretch":
        return np.clip(boxes, 0.0, 1.0)

    input_h = float(transform["input_h"])
    input_w = float(transform["input_w"])
    pad_top = float(transform["pad_top"])
    pad_left = float(transform["pad_left"])
    scale_y = max(1e-6, float(transform["scale_y"]))
    scale_x = max(1e-6, float(transform["scale_x"]))
    orig_h = max(1e-6, float(transform["orig_h"]))
    orig_w = max(1e-6, float(transform["orig_w"]))

    remapped = boxes.copy()
    remapped[:, [0, 2]] = ((remapped[:, [0, 2]] * input_h) - pad_top) / scale_y / orig_h
    remapped[:, [1, 3]] = ((remapped[:, [1, 3]] * input_w) - pad_left) / scale_x / orig_w
    return np.clip(remapped, 0.0, 1.0)


def resolve_raw_decode_profile(model_path):
    model_text = str(model_path).lower()
    if any(
        marker in model_text
        for marker in (
            "best_v11_b32_raw_uint8",
            "helmet3x_b8_raw_uint8",
            "stable_v3",
        )
    ):
        return "raw_uint8_direct"
    return "legacy"


def generate_anchors(input_h, input_w, min_level=3, max_level=7, anchor_scale=ANCHOR_SCALE):
    anchors = []
    for level in range(min_level, max_level + 1):
        stride = 2**level
        feat_h = int(math.ceil(input_h / stride))
        feat_w = int(math.ceil(input_w / stride))
        for y in range(feat_h):
            y_center = (y + 0.5) * stride / input_h
            for x in range(feat_w):
                x_center = (x + 0.5) * stride / input_w
                for scale_octave in range(NUM_SCALES):
                    octave_scale = scale_octave / NUM_SCALES
                    base_h = anchor_scale * stride * (2**octave_scale) / input_h
                    base_w = anchor_scale * stride * (2**octave_scale) / input_w
                    for aspect in ASPECT_RATIOS:
                        aspect_sqrt = math.sqrt(aspect)
                        anchors.append(
                            [y_center, x_center, base_h / aspect_sqrt, base_w * aspect_sqrt]
                        )
    return np.array(anchors, dtype=np.float32)


def dequantize_output(tensor, details):
    scale, zero_point = details.get("quantization", (0.0, 0))
    tensor = tensor.astype(np.float32)
    if scale:
        tensor = (tensor - zero_point) * scale
    return tensor


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def decode_boxes(raw_boxes, anchors, box_scale=BOX_SCALE, exp_clip=None):
    y_scale, x_scale, h_scale, w_scale = box_scale
    ty, tx, th, tw = [raw_boxes[..., i] for i in range(4)]
    ya, xa, ha, wa = [anchors[..., i] for i in range(4)]

    ycenter = ty / y_scale * ha + ya
    xcenter = tx / x_scale * wa + xa
    th_term = th / h_scale
    tw_term = tw / w_scale
    if exp_clip is not None:
        th_term = np.clip(th_term, -exp_clip, exp_clip)
        tw_term = np.clip(tw_term, -exp_clip, exp_clip)
    h = np.exp(th_term) * ha
    w = np.exp(tw_term) * wa

    ymin = ycenter - h / 2
    xmin = xcenter - w / 2
    ymax = ycenter + h / 2
    xmax = xcenter + w / 2
    return np.stack([ymin, xmin, ymax, xmax], axis=-1)


def iou(box, boxes):
    ymin = np.maximum(box[0], boxes[:, 0])
    xmin = np.maximum(box[1], boxes[:, 1])
    ymax = np.minimum(box[2], boxes[:, 2])
    xmax = np.minimum(box[3], boxes[:, 3])
    inter = np.maximum(0.0, ymax - ymin) * np.maximum(0.0, xmax - xmin)
    area1 = np.maximum(0.0, box[2] - box[0]) * np.maximum(0.0, box[3] - box[1])
    area2 = np.maximum(0.0, boxes[:, 2] - boxes[:, 0]) * np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-8
    return inter / union


def nms(boxes, scores, iou_threshold=NMS_IOU_THRESHOLD, max_det=MAX_DETECTIONS):
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0 and len(keep) < max_det:
        current = order[0]
        keep.append(current)
        if order.size == 1:
            break
        overlaps = iou(boxes[current], boxes[order[1:]])
        order = order[1:][overlaps < iou_threshold]
    return keep


def decode_raw_predictions(class_logits, box_deltas, anchors, raw_profile=None):
    raw_profile = raw_profile or RAW_DECODE_PROFILES["legacy"]

    if class_logits.min() < -0.01 or class_logits.max() > 1.01:
        class_scores = sigmoid(class_logits)
    else:
        class_scores = class_logits

    boxes = decode_boxes(
        box_deltas,
        anchors,
        box_scale=raw_profile.get("box_scale", BOX_SCALE),
        exp_clip=raw_profile.get("exp_clip"),
    )
    boxes = np.clip(boxes, 0.0, 1.0)

    final_boxes = []
    final_scores = []
    final_classes = []

    for class_id in range(class_scores.shape[1]):
        scores = class_scores[:, class_id]
        selected = np.where(scores >= CONF_THRESHOLD)[0]
        if selected.size == 0:
            continue
        keep = nms(boxes[selected], scores[selected])
        for keep_idx in keep:
            anchor_idx = selected[keep_idx]
            final_boxes.append(boxes[anchor_idx])
            final_scores.append(scores[anchor_idx])
            final_classes.append(class_id)

    if not final_scores:
        return (
            np.empty((0, 4), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    order = np.argsort(np.array(final_scores))[::-1][:MAX_DETECTIONS]
    ordered_boxes = np.array(final_boxes, dtype=np.float32)[order]
    ordered_classes = np.array(final_classes, dtype=np.int32)[order]
    ordered_scores = np.array(final_scores, dtype=np.float32)[order]
    return ordered_boxes, ordered_classes, ordered_scores


def decode_raw_outputs(output_details, raw_tensors, anchors, raw_profile=None):
    class_logits = dequantize_output(raw_tensors[0][0], output_details[0])
    box_deltas = dequantize_output(raw_tensors[1][0], output_details[1])
    return decode_raw_predictions(class_logits, box_deltas, anchors, raw_profile=raw_profile)

def decode_multiscale_raw_outputs(output_details, raw_tensors, anchors, raw_profile=None):
    class_channels = 9 * len(LABELS)
    box_channels = 9 * 4
    class_parts = {}
    box_parts = {}

    for details, tensor in zip(output_details, raw_tensors):
        shape = tuple(int(v) for v in details["shape"])
        if len(shape) != 4:
            continue
        size_key = (shape[1], shape[2])
        dequantized = dequantize_output(tensor[0], details)
        if shape[-1] == class_channels:
            class_parts[size_key] = dequantized.reshape(-1, len(LABELS))
        elif shape[-1] == box_channels:
            box_parts[size_key] = dequantized.reshape(-1, 4)

    if not class_parts or sorted(class_parts) != sorted(box_parts):
        raise RuntimeError(
            f"Unexpected multiscale raw output layout: classes={sorted(class_parts)} boxes={sorted(box_parts)}"
        )

    ordered_sizes = sorted(class_parts.keys(), key=lambda size: size[0], reverse=True)
    class_logits = np.concatenate([class_parts[size] for size in ordered_sizes], axis=0)
    box_deltas = np.concatenate([box_parts[size] for size in ordered_sizes], axis=0)

    if len(class_logits) != len(anchors) or len(box_deltas) != len(anchors):
        raise RuntimeError(
            f"Anchor mismatch for multiscale raw decode: cls={len(class_logits)} box={len(box_deltas)} anchors={len(anchors)}"
        )

    return decode_raw_predictions(class_logits, box_deltas, anchors, raw_profile=raw_profile)


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def clip_normalized_box(box):
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    return np.array(
        [
            clamp(ymin, 0.0, 1.0),
            clamp(xmin, 0.0, 1.0),
            clamp(ymax, 0.0, 1.0),
            clamp(xmax, 0.0, 1.0),
        ],
        dtype=np.float32,
    )


def expand_normalized_box(box, pad_x=0.0, pad_top=0.0, pad_bottom=0.0):
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    width = max(1e-6, xmax - xmin)
    height = max(1e-6, ymax - ymin)
    return clip_normalized_box(
        [
            ymin - height * pad_top,
            xmin - width * pad_x,
            ymax + height * pad_bottom,
            xmax + width * pad_x,
        ]
    )


def shrink_normalized_box(box, shrink_x=0.0, shrink_top=0.0, shrink_bottom=0.0):
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    width = max(1e-6, xmax - xmin)
    height = max(1e-6, ymax - ymin)
    return clip_normalized_box(
        [
            ymin + height * shrink_top,
            xmin + width * shrink_x,
            ymax - height * shrink_bottom,
            xmax - width * shrink_x,
        ]
    )


def make_square_crop(box, image_size, scale=1.25):
    img_w, img_h = image_size
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    x1 = xmin * img_w
    y1 = ymin * img_h
    x2 = xmax * img_w
    y2 = ymax * img_h

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    side = max(x2 - x1, y2 - y1) * scale
    side = max(side, 32.0)

    crop_x1 = int(round(cx - side / 2.0))
    crop_y1 = int(round(cy - side / 2.0))
    crop_x2 = int(round(cx + side / 2.0))
    crop_y2 = int(round(cy + side / 2.0))
    return crop_x1, crop_y1, crop_x2, crop_y2


def crop_with_padding(image, crop_box):
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_box
    img_w, img_h = image.size
    side = max(crop_x2 - crop_x1, crop_y2 - crop_y1)

    canvas = Image.new("RGB", (side, side))
    src_x1 = clamp(crop_x1, 0, img_w)
    src_y1 = clamp(crop_y1, 0, img_h)
    src_x2 = clamp(crop_x2, 0, img_w)
    src_y2 = clamp(crop_y2, 0, img_h)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return canvas, (crop_x1, crop_y1, side)

    crop = image.crop((src_x1, src_y1, src_x2, src_y2))
    paste_x = src_x1 - crop_x1
    paste_y = src_y1 - crop_y1
    canvas.paste(crop, (paste_x, paste_y))
    return canvas, (crop_x1, crop_y1, side)


def prepare_pose_input(image, input_details, input_size):
    input_h, input_w = input_size
    arr = np.array(image.resize((input_w, input_h), Image.BILINEAR))
    dtype = input_details[0]["dtype"]

    if dtype == np.float32:
        arr = arr.astype(np.float32)
    elif dtype == np.int32:
        arr = arr.astype(np.int32)
    else:
        arr = arr.astype(dtype)

    return np.expand_dims(arr, axis=0)


def parse_pose_output(output_tensor, output_details):
    output = dequantize_output(output_tensor, output_details[0])
    if output.ndim == 4:
        output = output[0, 0]
    elif output.ndim == 3:
        output = output[0]
    return output.astype(np.float32)


def run_pose_model(pose_model, image, box, crop_scale=1.25):
    crop_box = make_square_crop(box, image.size, scale=crop_scale)
    crop_img, crop_meta = crop_with_padding(image, crop_box)
    input_data = prepare_pose_input(crop_img, pose_model["input_details"], pose_model["input_size"])

    interp = pose_model["interp"]
    interp.set_tensor(pose_model["input_details"][0]["index"], input_data)
    interp.invoke()
    output_tensor = interp.get_tensor(pose_model["output_details"][0]["index"])
    keypoints = parse_pose_output(output_tensor, pose_model["output_details"])

    crop_x1, crop_y1, side = crop_meta
    mapped = []
    img_w, img_h = image.size
    for kp_y, kp_x, kp_score in keypoints:
        abs_x = crop_x1 + kp_x * side
        abs_y = crop_y1 + kp_y * side
        mapped.append(
            [
                np.clip(abs_y / img_h, 0.0, 1.0),
                np.clip(abs_x / img_w, 0.0, 1.0),
                float(kp_score),
            ]
        )
    return np.array(mapped, dtype=np.float32)


def estimate_poses(image, person_views, pose_model, args):
    if pose_model is None:
        return []

    candidates = []
    for idx, person_view in enumerate(person_views):
        if float(person_view["score"]) < args.pose_person_threshold:
            continue
        if not args.pose_all_persons and not person_view["missing"]:
            continue
        candidates.append((float(person_view["score"]), idx, person_view["box"]))

    candidates.sort(key=lambda item: item[0], reverse=True)
    poses = []
    for _, idx, box in candidates[: args.pose_max_persons]:
        poses.append((idx, run_pose_model(pose_model, image, box, crop_scale=args.pose_crop_scale)))
    return poses


def attach_poses_to_person_views(person_views, indexed_poses):
    if not indexed_poses:
        return person_views

    pose_map = {idx: pose for idx, pose in indexed_poses}
    enriched = []
    for idx, person_view in enumerate(person_views):
        updated = dict(person_view)
        updated["pose"] = pose_map.get(idx)
        enriched.append(updated)
    return enriched


def attach_cached_poses_to_person_views(person_views, cached_pose_views, min_iou=POSE_CACHE_MIN_IOU):
    if not cached_pose_views:
        return person_views

    enriched = []
    used = set()
    for person_view in person_views:
        best_idx = None
        best_score = min_iou
        for idx, cached in enumerate(cached_pose_views):
            if idx in used:
                continue
            score = float(iou(np.array(person_view["box"], dtype=np.float32), np.array([cached["box"]]))[0])
            if score > best_score:
                best_score = score
                best_idx = idx

        updated = dict(person_view)
        updated["pose"] = cached_pose_views[best_idx]["pose"] if best_idx is not None else None
        if best_idx is not None:
            used.add(best_idx)
        enriched.append(updated)

    return enriched


def count_pose_points(person_view, keypoint_ids, score_threshold=0.2):
    pose = person_view.get("pose")
    if pose is None:
        return 0
    count = 0
    for idx in keypoint_ids:
        if idx < len(pose) and float(pose[idx][2]) >= score_threshold:
            count += 1
    return count


def count_visible_pose_points(person_view, score_threshold=0.2):
    pose = person_view.get("pose")
    if pose is None:
        return 0
    return sum(1 for _, _, kp_score in pose if float(kp_score) >= score_threshold)


def get_person_box_metrics(box):
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)
    area = width * height
    aspect_ratio = height / max(width, 1e-6)
    return width, height, area, aspect_ratio


def has_person_shape_geometry(box):
    width, height, area, aspect_ratio = get_person_box_metrics(box)
    if aspect_ratio < PERSON_SHAPE_MIN_ASPECT_RATIO:
        return False
    if height < PERSON_SHAPE_MIN_HEIGHT and area < PERSON_SHAPE_MIN_AREA:
        return False
    if width > PERSON_SHAPE_MAX_WIDTH and aspect_ratio < PERSON_SHAPE_WIDE_MIN_ASPECT_RATIO:
        return False
    return True


def is_plausible_person_record(person):
    score = float(person["score"])
    width, height, area, aspect_ratio = get_person_box_metrics(person["box"])
    if aspect_ratio < 0.95:
        return False
    if width > PERSON_SHAPE_MAX_WIDTH and aspect_ratio < PERSON_SHAPE_WIDE_MIN_ASPECT_RATIO and score < PERSON_SHAPE_NO_POSE_MIN_SCORE:
        return False
    if height < (PERSON_SHAPE_MIN_HEIGHT * 0.75) and area < (PERSON_SHAPE_MIN_AREA * 0.75) and score < PERSON_SHAPE_NO_POSE_MIN_SCORE:
        return False
    return True


def should_suppress_partial_person(person_view, score_threshold=0.2):
    ymin, xmin, ymax, xmax = [float(v) for v in person_view["box"]]
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)
    area = width * height
    aspect_ratio = height / max(width, 1e-6)
    touches_side_edge = xmin <= PARTIAL_PERSON_EDGE_MARGIN or xmax >= (1.0 - PARTIAL_PERSON_EDGE_MARGIN)
    touches_top_edge = ymin <= PARTIAL_PERSON_EDGE_MARGIN
    oversized = (
        area >= PARTIAL_PERSON_MIN_AREA
        or width >= PARTIAL_PERSON_MIN_WIDTH
        or height >= PARTIAL_PERSON_MIN_HEIGHT
    )
    strict_edge_partial = touches_side_edge and (
        area >= PARTIAL_PERSON_STRICT_AREA
        or width >= PARTIAL_PERSON_STRICT_WIDTH
        or height >= PARTIAL_PERSON_STRICT_HEIGHT
    )
    flat_edge_box = touches_side_edge and aspect_ratio < PARTIAL_PERSON_MIN_ASPECT_RATIO
    pose_score_threshold = max(score_threshold, PARTIAL_PERSON_POSE_SCORE_THRESHOLD)
    head_points = count_pose_points(person_view, POSE_HEAD_KEYPOINT_IDS, score_threshold=pose_score_threshold)
    shoulder_points = count_pose_points(person_view, POSE_SHOULDER_KEYPOINT_IDS, score_threshold=pose_score_threshold)
    hip_points = count_pose_points(person_view, POSE_HIP_KEYPOINT_IDS, score_threshold=pose_score_threshold)
    torso_points = count_pose_points(person_view, POSE_TORSO_KEYPOINT_IDS, score_threshold=pose_score_threshold)

    if person_view.get("pose") is not None:
        has_head_evidence = head_points >= 1
        has_torso_evidence = torso_points >= 2
        has_upper_body_evidence = has_head_evidence and shoulder_points >= 1
        has_full_body_evidence = shoulder_points >= 1 and hip_points >= 1
        has_body_evidence = has_head_evidence or has_torso_evidence
        if flat_edge_box and not has_upper_body_evidence:
            return True
        if strict_edge_partial and not has_torso_evidence:
            return True
        if touches_side_edge and not (has_upper_body_evidence and has_full_body_evidence):
            return True
        if oversized and (touches_side_edge or touches_top_edge) and not (has_head_evidence and has_torso_evidence):
            return True
        if not has_body_evidence and (touches_side_edge or touches_top_edge or oversized):
            return True
    else:
        if flat_edge_box:
            return True
        if strict_edge_partial:
            return True
        if oversized and (touches_side_edge or touches_top_edge):
            return True

    return False


def suppress_partial_person_views(person_views, score_threshold=0.2):
    filtered = []
    for person_view in person_views:
        if should_suppress_partial_person(person_view, score_threshold=score_threshold):
            continue
        filtered.append(person_view)
    return filter_person_views_by_overlap(filtered)


def person_has_body_evidence(person_view, score_threshold=PARTIAL_PERSON_POSE_SCORE_THRESHOLD):
    head_points = count_pose_points(person_view, POSE_HEAD_KEYPOINT_IDS, score_threshold=score_threshold)
    shoulder_points = count_pose_points(person_view, POSE_SHOULDER_KEYPOINT_IDS, score_threshold=score_threshold)
    hip_points = count_pose_points(person_view, POSE_HIP_KEYPOINT_IDS, score_threshold=score_threshold)
    torso_points = count_pose_points(person_view, POSE_TORSO_KEYPOINT_IDS, score_threshold=score_threshold)
    return (head_points >= 1 and shoulder_points >= 1) or torso_points >= 2 or (shoulder_points >= 1 and hip_points >= 1)


def person_has_strong_body_evidence(person_view, score_threshold=PERSON_SHAPE_STRONG_POSE_SCORE_THRESHOLD):
    visible_points = count_visible_pose_points(person_view, score_threshold=score_threshold)
    head_points = count_pose_points(person_view, POSE_HEAD_KEYPOINT_IDS, score_threshold=score_threshold)
    shoulder_points = count_pose_points(person_view, POSE_SHOULDER_KEYPOINT_IDS, score_threshold=score_threshold)
    hip_points = count_pose_points(person_view, POSE_HIP_KEYPOINT_IDS, score_threshold=score_threshold)
    torso_points = count_pose_points(person_view, POSE_TORSO_KEYPOINT_IDS, score_threshold=score_threshold)
    return (
        (torso_points >= 3 and visible_points >= PERSON_SHAPE_MIN_VISIBLE_KEYPOINTS)
        or (shoulder_points == 2 and hip_points >= 1 and visible_points >= PERSON_SHAPE_MIN_VISIBLE_KEYPOINTS)
        or (head_points >= 1 and shoulder_points == 2 and visible_points >= PERSON_SHAPE_MIN_VISIBLE_KEYPOINTS)
    )


def should_suppress_nonhuman_person(person_view, score_threshold=PARTIAL_PERSON_POSE_SCORE_THRESHOLD):
    score = float(person_view["score"])
    pose_score_threshold = max(score_threshold, PERSON_SHAPE_STRONG_POSE_SCORE_THRESHOLD)
    if person_has_strong_body_evidence(person_view, score_threshold=pose_score_threshold):
        return False

    has_geometry = has_person_shape_geometry(person_view["box"])
    visible_points = count_visible_pose_points(person_view, score_threshold=pose_score_threshold)
    head_points = count_pose_points(person_view, POSE_HEAD_KEYPOINT_IDS, score_threshold=pose_score_threshold)
    shoulder_points = count_pose_points(person_view, POSE_SHOULDER_KEYPOINT_IDS, score_threshold=pose_score_threshold)
    hip_points = count_pose_points(person_view, POSE_HIP_KEYPOINT_IDS, score_threshold=pose_score_threshold)
    torso_points = count_pose_points(person_view, POSE_TORSO_KEYPOINT_IDS, score_threshold=pose_score_threshold)

    if person_view.get("pose") is not None:
        has_partial_pose = (
            visible_points >= PERSON_SHAPE_MIN_VISIBLE_KEYPOINTS
            and torso_points >= 2
            and (shoulder_points >= 1 or hip_points >= 1)
        )
        if has_partial_pose and has_geometry and score >= PERSON_SHAPE_POSE_MIN_SCORE:
            return False
        return True

    return (not has_geometry) or score < PERSON_SHAPE_NO_POSE_MIN_SCORE


def suppress_nonhuman_person_views(person_views, score_threshold=PARTIAL_PERSON_POSE_SCORE_THRESHOLD):
    filtered = []
    for person_view in person_views:
        if should_suppress_nonhuman_person(person_view, score_threshold=score_threshold):
            continue
        filtered.append(person_view)
    return filter_person_views_by_overlap(filtered)


def should_track_recent_person(person_view):
    ymin, xmin, ymax, xmax = [float(v) for v in person_view["box"]]
    width = max(0.0, xmax - xmin)
    height = max(0.0, ymax - ymin)
    area = width * height
    aspect_ratio = height / max(width, 1e-6)
    touches_edge = (
        xmin <= PARTIAL_PERSON_EDGE_MARGIN
        or xmax >= (1.0 - PARTIAL_PERSON_EDGE_MARGIN)
        or ymin <= PARTIAL_PERSON_EDGE_MARGIN
    )
    if area < STABLE_PERSON_MIN_AREA or aspect_ratio < PARTIAL_PERSON_MIN_ASPECT_RATIO:
        return False
    if touches_edge and not person_has_body_evidence(person_view):
        return False
    return person_has_body_evidence(person_view) or float(person_view["score"]) >= STABLE_PERSON_MIN_SCORE


def should_keep_recent_person(person_view, hits):
    if not should_track_recent_person(person_view):
        return False
    if hits < STABLE_PERSON_MIN_HITS:
        return False
    return True


def stabilize_person_views(person_views, recent_person_views, frame_count):
    merged = [{**dict(person_view), "stabilized": False} for person_view in person_views]
    next_recent = []
    used_recent = set()

    for person_view in person_views:
        best_idx = None
        best_score = STABLE_PERSON_MIN_IOU
        for idx, recent in enumerate(recent_person_views):
            if idx in used_recent:
                continue
            score = float(
                iou(
                    np.array(person_view["box"], dtype=np.float32),
                    np.array([recent["box"]], dtype=np.float32),
                )[0]
            )
            if score > best_score:
                best_score = score
                best_idx = idx

        hits = 1
        if best_idx is not None:
            used_recent.add(best_idx)
            hits = int(recent_person_views[best_idx]["hits"]) + 1

        if should_track_recent_person(person_view):
            next_recent.append(
                {
                    "box": np.array(person_view["box"], dtype=np.float32),
                    "view": {**dict(person_view), "stabilized": False},
                    "last_seen_frame": frame_count,
                    "hits": hits,
                }
            )

    for idx, recent in enumerate(recent_person_views):
        if idx in used_recent:
            continue
        age = frame_count - int(recent["last_seen_frame"])
        if age <= 0 or age > STABLE_PERSON_HOLD_FRAMES:
            continue

        recent_view = dict(recent["view"])
        if not should_keep_recent_person(recent_view, int(recent["hits"])):
            continue

        duplicate = False
        for person_view in merged:
            overlap = float(
                iou(
                    np.array(recent_view["box"], dtype=np.float32),
                    np.array([person_view["box"]], dtype=np.float32),
                )[0]
            )
            if overlap >= STABLE_PERSON_MIN_IOU:
                duplicate = True
                break
        if duplicate:
            continue

        recent_view["score"] = float(recent_view["score"]) * (STABLE_PERSON_DECAY ** age)
        recent_view["stabilized"] = True
        merged.append(recent_view)
        next_recent.append(recent)

    merged = filter_person_views_by_overlap(merged)
    return merged, next_recent


def should_run_pose_for_person_views(person_views, frame_count, args):
    return (
        (frame_count % max(1, args.pose_every)) == 0
        and any(
            float(person_view["score"]) >= args.pose_person_threshold
            and (args.pose_all_persons or person_view["missing"])
            for person_view in person_views
        )
    )


def build_cached_pose_views(person_views):
    return [
        {"box": np.array(person_view["box"], dtype=np.float32), "pose": person_view["pose"]}
        for person_view in person_views
        if person_view.get("pose") is not None
    ]


def attach_pose_views(image_or_frame, person_views, pose_model, args, frame_count, cached_pose_views):
    if pose_model is None:
        person_views = suppress_partial_person_views(person_views)
        person_views = suppress_nonhuman_person_views(person_views)
        return person_views, cached_pose_views

    if should_run_pose_for_person_views(person_views, frame_count, args):
        image = ensure_pil_image(image_or_frame)
        indexed_poses = estimate_poses(image, person_views, pose_model, args)
        person_views = attach_poses_to_person_views(person_views, indexed_poses)
        cached_pose_views = build_cached_pose_views(person_views)
    else:
        person_views = attach_cached_poses_to_person_views(person_views, cached_pose_views)

    person_views = suppress_partial_person_views(person_views)
    person_views = suppress_nonhuman_person_views(person_views, score_threshold=args.pose_score_threshold)
    return person_views, cached_pose_views


def rematch_person_views_with_pose(person_views, boxes, classes, scores):
    helmets, vests = collect_item_records(boxes, classes, scores)
    used_helmets = collect_used_item_indices(person_views, helmets, "helmet_box")
    used_vests = collect_used_item_indices(person_views, vests, "vest_box")
    updated_views = []

    for person_view in person_views:
        updated = dict(person_view)
        if updated.get("pose") is not None:
            if not updated["has_helmet"]:
                best_idx, best_score = find_best_item_match(updated, helmets, "helmet", used_indices=used_helmets)
                if best_idx is not None and best_score >= HELMET_MATCH_SCORE_MIN:
                    updated["has_helmet"] = True
                    updated["helmet_box"] = helmets[best_idx]["box"]
                    updated["helmet_score"] = helmets[best_idx]["score"]
                    updated.setdefault("match_scores", {})["helmet"] = best_score
                    used_helmets.add(best_idx)

            if not updated["has_vest"]:
                best_idx, best_score = find_best_item_match(updated, vests, "vest", used_indices=used_vests)
                if best_idx is not None and best_score >= VEST_MATCH_SCORE_MIN:
                    updated["has_vest"] = True
                    updated["vest_box"] = vests[best_idx]["box"]
                    updated["vest_score"] = vests[best_idx]["score"]
                    updated.setdefault("match_scores", {})["vest"] = best_score
                    used_vests.add(best_idx)

        missing = []
        if not updated["has_helmet"]:
            missing.append("NO_HELMET")
        if not updated["has_vest"]:
            missing.append("NO_VEST")
        updated["missing"] = missing
        updated_views.append(updated)

    return updated_views


def collect_pose_arrays(person_views):
    return [person_view["pose"] for person_view in person_views if person_view.get("pose") is not None]


def compute_head_alert_rect(person_view, img_w, img_h, score_threshold=0.2):
    pose = person_view.get("pose")
    if pose is not None:
        points = []
        for idx in POSE_HEAD_KEYPOINT_IDS:
            if idx < len(pose) and float(pose[idx][2]) >= score_threshold:
                points.append((float(pose[idx][1]) * img_w, float(pose[idx][0]) * img_h))

        if points:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            shoulder_points = []
            for idx in POSE_SHOULDER_KEYPOINT_IDS:
                if idx < len(pose) and float(pose[idx][2]) >= score_threshold:
                    shoulder_points.append((float(pose[idx][1]) * img_w, float(pose[idx][0]) * img_h))

            person_ymin, person_xmin, person_ymax, person_xmax = [float(v) for v in person_view["box"]]
            person_w = max(1.0, (person_xmax - person_xmin) * img_w)
            person_h = max(1.0, (person_ymax - person_ymin) * img_h)
            shoulder_span = (
                abs(shoulder_points[0][0] - shoulder_points[1][0])
                if len(shoulder_points) == 2
                else person_w * 0.28
            )

            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            half_w = max(12, int(max((max_x - min_x) * 1.10, shoulder_span * 0.30, person_w * 0.11)))
            half_h = max(14, int(max((max_y - min_y) * 1.45, person_h * 0.10, shoulder_span * 0.22)))
            center_x = int((min_x + max_x) / 2.0)
            upward_shift = max(8, int(max((max_y - min_y) * 0.95, person_h * 0.05, shoulder_span * 0.12)))
            center_y = int((min_y + max_y) / 2.0) - upward_shift
            return (
                center_x - half_w,
                center_y - half_h,
                center_x + half_w,
                center_y + half_h,
            )

    return compute_head_alert_rect_b0(person_view, img_w, img_h)


def compute_vest_alert_rect(person_view, img_w, img_h, score_threshold=0.2):
    pose = person_view.get("pose")
    if pose is not None:
        points = []
        for idx in (5, 6, 11, 12):
            if idx < len(pose) and float(pose[idx][2]) >= score_threshold:
                points.append((float(pose[idx][1]) * img_w, float(pose[idx][0]) * img_h))
        if len(points) >= 2:
            xs = [p[0] for p in points]
            ys = [p[1] for p in points]
            pad_x = max(12, int((max(xs) - min(xs)) * 0.28))
            pad_y = max(14, int((max(ys) - min(ys)) * 0.28))
            return (
                int(min(xs) - pad_x),
                int(min(ys) - pad_y),
                int(max(xs) + pad_x),
                int(max(ys) + pad_y),
            )

    return compute_vest_alert_rect_b0(person_view, img_w, img_h)


def compute_head_alert_rect_b0(person_view, img_w, img_h):
    ymin, xmin, ymax, xmax = [float(v) for v in person_view["box"]]
    x1, y1 = int(xmin * img_w), int(ymin * img_h)
    x2, y2 = int(xmax * img_w), int(ymax * img_h)
    person_w = max(1, x2 - x1)
    person_h = max(1, y2 - y1)

    head_cx = int((x1 + x2) / 2)
    head_cy = int(y1 + person_h * 0.06)
    half_w = max(12, int(person_w * 0.30))
    half_h = max(14, int(person_h * 0.12))
    return (
        head_cx - half_w,
        head_cy - half_h,
        head_cx + half_w,
        head_cy + half_h,
    )


def compute_vest_alert_rect_b0(person_view, img_w, img_h):
    ymin, xmin, ymax, xmax = [float(v) for v in person_view["box"]]
    x1, y1 = int(xmin * img_w), int(ymin * img_h)
    x2, y2 = int(xmax * img_w), int(ymax * img_h)
    person_w = max(1, x2 - x1)
    person_h = max(1, y2 - y1)

    torso_cx = int((x1 + x2) / 2)
    torso_cy = int(y1 + person_h * 0.42)
    half_w = max(14, int(person_w * 0.35))
    half_h = max(14, int(person_h * 0.12))
    return (
        torso_cx - half_w,
        torso_cy - half_h,
        torso_cx + half_w,
        torso_cy + half_h,
    )


def draw_pose(image, poses, score_threshold):
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size

    for pose in poses:
        for (start_idx, end_idx), color in zip(POSE_EDGES, POSE_EDGE_COLORS):
            if pose[start_idx][2] < score_threshold or pose[end_idx][2] < score_threshold:
                continue
            x1 = int(pose[start_idx][1] * img_w)
            y1 = int(pose[start_idx][0] * img_h)
            x2 = int(pose[end_idx][1] * img_w)
            y2 = int(pose[end_idx][0] * img_h)
            draw.line((x1, y1, x2, y2), fill=color, width=3)

        for idx, (kp_y, kp_x, kp_score) in enumerate(pose):
            if kp_score < score_threshold:
                continue
            x = int(kp_x * img_w)
            y = int(kp_y * img_h)
            radius = 4
            draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=(255, 64, 64))
            if idx == 0:
                draw.text((x + 6, y - 6), "nose", fill=(255, 255, 255))

    return image


def run_inference(interp, input_details, output_details, image, input_size, model_info):
    input_data, input_transform = prepare_detector_input(
        image,
        input_details,
        input_size,
        resize_mode=model_info.get("resize_mode", "letterbox"),
    )

    t0 = time.perf_counter()
    if model_info.get("runtime") == "edgetpu_dual":
        # PyCoral's pipeline runner returns each float32 tensor in a padded 16-byte layout.
        # Sampling every 4th float reconstructs the same outputs as the single-TPU model.
        interp["runner"].push({interp["input_name"]: input_data})
        raw_outputs = interp["runner"]._runner.Pop()
    else:
        interp.set_tensor(input_details[0]["index"], input_data)
        interp.invoke()
    latency_ms = (time.perf_counter() - t0) * 1000

    if model_info.get("runtime") == "edgetpu_dual":
        stride = int(model_info.get("dual_output_stride", 1))
        tensors_by_name = {}
        for details in output_details:
            flat = raw_outputs[details["name"]][::stride]
            tensors_by_name[details["name"]] = flat.reshape(details["shape"])

        boxes = tensors_by_name["StatefulPartitionedCall:3"][0]
        classes = tensors_by_name["StatefulPartitionedCall:2"][0]
        scores = tensors_by_name["StatefulPartitionedCall:1"][0]
        count = int(tensors_by_name["StatefulPartitionedCall:0"].reshape(-1)[0])
        boxes = boxes[:count]
        classes = classes[:count]
        scores = scores[:count]
    elif model_info["is_raw_model"]:
        raw_tensors = [interp.get_tensor(details["index"]) for details in output_details]
        if model_info.get("raw_layout") == "multiscale":
            boxes, classes, scores = decode_multiscale_raw_outputs(
                output_details,
                raw_tensors,
                model_info["anchors"],
                raw_profile=model_info.get("raw_profile"),
            )
        else:
            boxes, classes, scores = decode_raw_outputs(
                output_details,
                raw_tensors,
                model_info["anchors"],
                raw_profile=model_info.get("raw_profile"),
            )
    else:
        scores = dequantize_output(interp.get_tensor(output_details[0]["index"]), output_details[0])[0]
        boxes = dequantize_output(interp.get_tensor(output_details[1]["index"]), output_details[1])[0]
        count_tensor = dequantize_output(interp.get_tensor(output_details[2]["index"]), output_details[2])
        classes = dequantize_output(interp.get_tensor(output_details[3]["index"]), output_details[3])[0]
        count = int(np.rint(np.reshape(count_tensor, (-1,))[0]))
        classes = np.rint(classes).astype(np.int32)
        boxes = np.clip(boxes, 0.0, 1.0)
        boxes = boxes[:count]
        classes = classes[:count]
        scores = scores[:count]

    boxes = remap_boxes_to_original(boxes, input_transform)
    return boxes, classes, scores, latency_ms


def box_center(box):
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    return (ymin + ymax) / 2.0, (xmin + xmax) / 2.0


def box_area(box):
    ymin, xmin, ymax, xmax = [float(v) for v in box]
    return max(0.0, ymax - ymin) * max(0.0, xmax - xmin)


def intersection_area(box_a, box_b):
    aymin, axmin, aymax, axmax = [float(v) for v in box_a]
    bymin, bxmin, bymax, bxmax = [float(v) for v in box_b]
    inter_ymin = max(aymin, bymin)
    inter_xmin = max(axmin, bxmin)
    inter_ymax = min(aymax, bymax)
    inter_xmax = min(axmax, bxmax)
    return max(0.0, inter_ymax - inter_ymin) * max(0.0, inter_xmax - inter_xmin)


def box_containment_ratio(box_a, box_b):
    inter = intersection_area(box_a, box_b)
    min_area = max(1e-6, min(box_area(box_a), box_area(box_b)))
    return inter / min_area


def filter_person_records(persons):
    kept = []
    for person in sorted(persons, key=lambda record: record["score"], reverse=True):
        if not is_plausible_person_record(person):
            continue
        duplicate = False
        for kept_person in kept:
            overlap = float(
                iou(
                    np.array(person["box"], dtype=np.float32),
                    np.array([kept_person["box"]], dtype=np.float32),
                )[0]
            )
            containment = box_containment_ratio(person["box"], kept_person["box"])
            if overlap >= PERSON_DUPLICATE_IOU_THRESHOLD or containment >= PERSON_DUPLICATE_CONTAINMENT_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append(person)
    return kept


def filter_person_views_by_overlap(person_views):
    kept = []
    for person_view in sorted(
        person_views,
        key=lambda view: (
            0 if view.get("stabilized") else 1,
            1 if view.get("missing") else 0,
            float(view["score"]),
        ),
        reverse=True,
    ):
        duplicate = False
        for kept_view in kept:
            overlap = float(
                iou(
                    np.array(person_view["box"], dtype=np.float32),
                    np.array([kept_view["box"]], dtype=np.float32),
                )[0]
            )
            containment = box_containment_ratio(person_view["box"], kept_view["box"])
            if overlap >= PERSON_DUPLICATE_IOU_THRESHOLD or containment >= PERSON_DUPLICATE_CONTAINMENT_THRESHOLD:
                duplicate = True
                break
        if not duplicate:
            kept.append(person_view)
    return kept


def compute_pose_guided_region(person_view, kind, score_threshold=0.2):
    pose = person_view.get("pose")
    if pose is None:
        return None

    pymin, pxmin, pymax, pxmax = [float(v) for v in person_view["box"]]
    p_h = max(1e-6, pymax - pymin)
    p_w = max(1e-6, pxmax - pxmin)

    def points_for(ids):
        pts = []
        for idx in ids:
            if idx < len(pose) and float(pose[idx][2]) >= score_threshold:
                pts.append((float(pose[idx][0]), float(pose[idx][1])))
        return pts

    if kind == "helmet":
        head_points = points_for(POSE_HEAD_KEYPOINT_IDS)
        shoulder_points = points_for(POSE_SHOULDER_KEYPOINT_IDS)
        if not head_points:
            return None
        head_ys = [p[0] for p in head_points]
        head_xs = [p[1] for p in head_points]
        shoulder_span = (
            abs(shoulder_points[0][1] - shoulder_points[1][1])
            if len(shoulder_points) == 2
            else p_w * 0.25
        )
        pad_x = max((max(head_xs) - min(head_xs)) * 0.45, shoulder_span * 0.20, p_w * 0.06)
        pad_top = max((max(head_ys) - min(head_ys)) * 0.80, p_h * 0.06)
        pad_bottom = max((max(head_ys) - min(head_ys)) * 0.60, p_h * 0.05)
        return (
            max(0.0, min(head_ys) - pad_top),
            max(0.0, min(head_xs) - pad_x),
            min(1.0, max(head_ys) + pad_bottom),
            min(1.0, max(head_xs) + pad_x),
        )

    shoulder_points = points_for(POSE_SHOULDER_KEYPOINT_IDS)
    hip_points = points_for(POSE_HIP_KEYPOINT_IDS)
    torso_points = shoulder_points + hip_points
    if len(torso_points) < 3:
        return None
    torso_ys = [p[0] for p in torso_points]
    torso_xs = [p[1] for p in torso_points]
    pad_x = max((max(torso_xs) - min(torso_xs)) * 0.18, p_w * 0.08)
    pad_y = max((max(torso_ys) - min(torso_ys)) * 0.12, p_h * 0.05)
    return (
        max(0.0, min(torso_ys) - pad_y),
        max(0.0, min(torso_xs) - pad_x),
        min(1.0, max(torso_ys) + pad_y),
        min(1.0, max(torso_xs) + pad_x),
    )


def ppe_match_score(person_view, item_box, kind, item_score=1.0):
    pymin, pxmin, pymax, pxmax = [float(v) for v in person_view["box"]]
    iymin, ixmin, iymax, ixmax = [float(v) for v in item_box]
    p_h = max(1e-6, pymax - pymin)
    p_w = max(1e-6, pxmax - pxmin)
    icy, icx = box_center(item_box)

    region = compute_pose_guided_region(person_view, kind, score_threshold=0.2)
    if region is None:
        x_margin_ratio = HELMET_MATCH_X_MARGIN if kind == "helmet" else VEST_MATCH_X_MARGIN
        x_margin = p_w * x_margin_ratio
        if not (pxmin - x_margin <= icx <= pxmax + x_margin):
            return -1.0

        if kind == "helmet":
            region_ymin = pymin + HELMET_REGION_TOP * p_h
            region_ymax = pymin + HELMET_REGION_BOTTOM * p_h
        else:
            region_ymin = pymin + VEST_REGION_TOP * p_h
            region_ymax = pymin + VEST_REGION_BOTTOM * p_h
        region_xmin = pxmin - x_margin
        region_xmax = pxmax + x_margin
    else:
        region_ymin, region_xmin, region_ymax, region_xmax = region

    if not (region_xmin <= icx <= region_xmax):
        return -1.0

    if not (region_ymin <= icy <= region_ymax):
        return -1.0

    overlap_person = iou(
        np.array([iymin, ixmin, iymax, ixmax], dtype=np.float32),
        np.array([[pymin, pxmin, pymax, pxmax]], dtype=np.float32),
    )[0]
    overlap_region = iou(
        np.array([iymin, ixmin, iymax, ixmax], dtype=np.float32),
        np.array([[region_ymin, region_xmin, region_ymax, region_xmax]], dtype=np.float32),
    )[0]
    overlap = max(float(overlap_person), float(overlap_region))

    region_cy = (region_ymin + region_ymax) / 2.0
    region_cx = (region_xmin + region_xmax) / 2.0
    diag = max(1e-6, math.sqrt(p_h ** 2 + p_w ** 2))
    dist = math.sqrt((icy - region_cy) ** 2 + (icx - region_cx) ** 2)
    d_norm = min(1.0, dist / diag)
    c_det = float(item_score)

    score = 0.45 * overlap + 0.35 * (1.0 - d_norm) + 0.20 * c_det
    return float(score)


def collect_item_records(boxes, classes, scores):
    helmets = []
    vests = []
    for i in range(len(scores)):
        cls_id = int(classes[i])
        label = LABELS.get(cls_id, f"cls{cls_id}")
        if label not in {"helmet", "vest"}:
            continue
        threshold = CONF_THRESHOLDS.get(label, CONF_THRESHOLD)
        if float(scores[i]) < threshold:
            continue
        record = {
            "class_id": cls_id,
            "label": label,
            "box": np.array(boxes[i], dtype=np.float32),
            "score": float(scores[i]),
        }
        if label == "helmet":
            helmets.append(record)
        else:
            vests.append(record)
    return helmets, vests


def find_best_item_match(person_view, items, kind, used_indices=None):
    used_indices = used_indices or set()
    best_idx = None
    best_score = -1.0
    for idx, item in enumerate(items):
        if idx in used_indices:
            continue
        score = ppe_match_score(person_view, item["box"], kind, item_score=item["score"])
        if score > best_score:
            best_idx = idx
            best_score = score
    return best_idx, best_score


def collect_used_item_indices(person_views, items, box_key, min_iou=0.5):
    used = set()
    for person_view in person_views:
        item_box = person_view.get(box_key)
        if item_box is None:
            continue
        best_idx = None
        best_score = min_iou
        for idx, item in enumerate(items):
            if idx in used:
                continue
            score = float(
                iou(
                    np.array(item_box, dtype=np.float32),
                    np.array([item["box"]], dtype=np.float32),
                )[0]
            )
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            used.add(best_idx)
    return used


def apply_ppe_matches(person_views, helmets, vests):
    used_helmets = set()
    used_vests = set()
    updated_views = []

    for person_view in person_views:
        best_helmet, best_helmet_score = find_best_item_match(person_view, helmets, "helmet", used_indices=used_helmets)
        best_vest, best_vest_score = find_best_item_match(person_view, vests, "vest", used_indices=used_vests)

        has_helmet = best_helmet is not None and best_helmet_score >= HELMET_MATCH_SCORE_MIN
        has_vest = best_vest is not None and best_vest_score >= VEST_MATCH_SCORE_MIN
        if has_helmet:
            used_helmets.add(best_helmet)
        if has_vest:
            used_vests.add(best_vest)

        missing = []
        if not has_helmet:
            missing.append("NO_HELMET")
        if not has_vest:
            missing.append("NO_VEST")

        updated_views.append(
            {
                **dict(person_view),
                "has_helmet": has_helmet,
                "has_vest": has_vest,
                "helmet_box": helmets[best_helmet]["box"] if has_helmet else None,
                "helmet_score": helmets[best_helmet]["score"] if has_helmet else None,
                "vest_box": vests[best_vest]["box"] if has_vest else None,
                "vest_score": vests[best_vest]["score"] if has_vest else None,
                "missing": missing,
                "match_scores": {"helmet": best_helmet_score, "vest": best_vest_score},
            }
        )

    return updated_views


def build_person_views(boxes, classes, scores):
    records = []
    for i in range(len(scores)):
        cls_id = int(classes[i])
        label = LABELS.get(cls_id, f"cls{cls_id}")
        threshold = CONF_THRESHOLDS.get(label, CONF_THRESHOLD)
        if float(scores[i]) < threshold:
            continue
        records.append(
            {
                "class_id": cls_id,
                "label": label,
                "box": np.array(boxes[i], dtype=np.float32),
                "score": float(scores[i]),
            }
        )

    persons = filter_person_records([r for r in records if r["label"] == "person"])
    helmets = [r for r in records if r["label"] == "helmet"]
    vests = [r for r in records if r["label"] == "vest"]
    base_person_views = [{"box": person["box"], "score": person["score"]} for person in persons]
    return apply_ppe_matches(base_person_views, helmets, vests)


def format_person_status(person_view):
    score_text = f"{person_view['score']:.0%}"
    return f"person {score_text}"


def get_person_display_box(person_view, score_threshold=0.2, paper_style=False):
    ymin, xmin, ymax, xmax = [float(v) for v in person_view["box"]]
    width = max(1e-6, xmax - xmin)
    height = max(1e-6, ymax - ymin)
    base_box = np.array([ymin, xmin, ymax, xmax], dtype=np.float32)
    is_violation = bool(person_view.get("missing"))

    pose = person_view.get("pose")
    if pose is not None:
        points = []
        for kp_y, kp_x, kp_score in pose:
            if float(kp_score) >= score_threshold:
                points.append((float(kp_y), float(kp_x)))

        if len(points) >= 5:
            ys = [p[0] for p in points]
            xs = [p[1] for p in points]
            if paper_style:
                pose_width = max(xs) - min(xs)
                pose_height = max(ys) - min(ys)
                pose_box = clip_normalized_box(
                    [
                        min(ys) - max(pose_height * 0.12, height * PAPER_PERSON_POSE_PAD_TOP),
                        min(xs) - max(pose_width * 0.10, width * PAPER_PERSON_POSE_PAD_X),
                        max(ys) + max(pose_height * 0.08, height * PAPER_PERSON_POSE_PAD_BOTTOM),
                        max(xs) + max(pose_width * 0.10, width * PAPER_PERSON_POSE_PAD_X),
                    ]
                )
                display_box = pose_box
                shrink_x = PAPER_VIOLATION_PERSON_SHRINK_X if is_violation else PAPER_COMPLIANT_PERSON_SHRINK_X
                shrink_top = PAPER_VIOLATION_PERSON_SHRINK_TOP if is_violation else PAPER_COMPLIANT_PERSON_SHRINK_TOP
                shrink_bottom = PAPER_VIOLATION_PERSON_SHRINK_BOTTOM if is_violation else PAPER_COMPLIANT_PERSON_SHRINK_BOTTOM
                display_box = shrink_normalized_box(
                    display_box,
                    shrink_x=shrink_x,
                    shrink_top=shrink_top,
                    shrink_bottom=shrink_bottom,
                )
                return expand_normalized_box(
                    display_box,
                    pad_x=PAPER_VIOLATION_PERSON_BOX_PAD_X if is_violation else PAPER_PERSON_BOX_PAD_X,
                    pad_top=PAPER_VIOLATION_PERSON_BOX_PAD_TOP if is_violation else PAPER_PERSON_BOX_PAD_TOP,
                    pad_bottom=PAPER_VIOLATION_PERSON_BOX_PAD_BOTTOM if is_violation else PAPER_PERSON_BOX_PAD_BOTTOM,
                )

            pad_x = max((max(xs) - min(xs)) * 0.18, width * PERSON_DISPLAY_POSE_PAD_X)
            pad_top = max((max(ys) - min(ys)) * 0.10, height * PERSON_DISPLAY_POSE_PAD_TOP)
            pad_bottom = max((max(ys) - min(ys)) * 0.12, height * PERSON_DISPLAY_POSE_PAD_BOTTOM)
            return clip_normalized_box(
                [
                    max(ymin, min(ys) - pad_top),
                    max(xmin, min(xs) - pad_x),
                    min(ymax, max(ys) + pad_bottom),
                    min(xmax, max(xs) + pad_x),
                ]
            )

    if paper_style:
        shrink_x = PAPER_VIOLATION_PERSON_SHRINK_X if is_violation else PAPER_COMPLIANT_PERSON_SHRINK_X
        shrink_top = PAPER_VIOLATION_PERSON_SHRINK_TOP if is_violation else PAPER_COMPLIANT_PERSON_SHRINK_TOP
        shrink_bottom = PAPER_VIOLATION_PERSON_SHRINK_BOTTOM if is_violation else PAPER_COMPLIANT_PERSON_SHRINK_BOTTOM
        paper_box = shrink_normalized_box(
            base_box,
            shrink_x=shrink_x,
            shrink_top=shrink_top,
            shrink_bottom=shrink_bottom,
        )
        return expand_normalized_box(
            paper_box,
            pad_x=PAPER_VIOLATION_PERSON_BOX_PAD_X if is_violation else PAPER_PERSON_BOX_PAD_X,
            pad_top=PAPER_VIOLATION_PERSON_BOX_PAD_TOP if is_violation else PAPER_PERSON_BOX_PAD_TOP,
            pad_bottom=PAPER_VIOLATION_PERSON_BOX_PAD_BOTTOM if is_violation else PAPER_PERSON_BOX_PAD_BOTTOM,
        )

    return clip_normalized_box(
        [
            ymin + height * PERSON_DISPLAY_SHRINK_TOP,
            xmin + width * PERSON_DISPLAY_SHRINK_X,
            ymax - height * PERSON_DISPLAY_SHRINK_BOTTOM,
            xmax - width * PERSON_DISPLAY_SHRINK_X,
        ]
    )


def draw_tagged_box_pil(draw, rect, label, color, font):
    x1, y1, x2, y2 = [int(v) for v in rect]
    draw.rectangle((x1, y1, x2, y2), outline=color, width=3)
    bbox = draw.textbbox((x1, y1 - 18), label, font=font)
    draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=color)
    draw.text((x1, y1 - 18), label, fill=(255, 255, 255), font=font)


def draw_tagged_box_cv(frame_bgr, rect, label, color_bgr):
    import cv2

    x1, y1, x2, y2 = [int(v) for v in rect]
    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), color_bgr, 3, cv2.LINE_AA)
    (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    text_y = max(text_h + 4, y1 - 6)
    cv2.rectangle(
        frame_bgr,
        (x1, text_y - text_h - baseline - 6),
        (x1 + text_w + 8, text_y),
        color_bgr,
        -1,
    )
    cv2.putText(
        frame_bgr,
        label,
        (x1 + 4, text_y - baseline - 3),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )


def draw_results(image, boxes, classes, scores, latency_ms, person_views=None, paper_style=False):
    draw = ImageDraw.Draw(image)
    img_w, img_h = image.size
    person_views = person_views if person_views is not None else build_person_views(boxes, classes, scores)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 14)
    except IOError:
        font = ImageFont.load_default()

    violations = 0
    for person_view in person_views:
        ymin, xmin, ymax, xmax = get_person_display_box(person_view, paper_style=paper_style)
        x1, y1 = int(xmin * img_w), int(ymin * img_h)
        x2, y2 = int(xmax * img_w), int(ymax * img_h)

        if person_view["missing"]:
            violations += 1
            color = ALERT_COLOR
            text = f"VIOLATION {person_view['score']:.0%}"
        else:
            color = (0, 160, 0)
            text = "" if paper_style else f"OK {person_view['score']:.0%}"

        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if text:
            bbox = draw.textbbox((x1, y1 - 18), text, font=font)
            draw.rectangle([bbox[0] - 1, bbox[1] - 1, bbox[2] + 1, bbox[3] + 1], fill=color)
            draw.text((x1, y1 - 18), text, fill=(255, 255, 255), font=font)

        if not person_view["has_helmet"]:
            draw_tagged_box_pil(
                draw,
                compute_head_alert_rect(person_view, img_w, img_h, score_threshold=0.2),
                "NoHelmet",
                ALERT_COLOR,
                font,
            )

        if not person_view["has_vest"]:
            draw_tagged_box_pil(
                draw,
                compute_vest_alert_rect(person_view, img_w, img_h, score_threshold=0.2),
                "NoVest",
                ALERT_COLOR,
                font,
            )

    info = f"{latency_ms:.0f}ms | {1000/latency_ms:.1f} FPS"
    draw.rectangle([0, 0, img_w, 22], fill=(0, 0, 0))
    draw.text((5, 3), info, fill=(255, 255, 255), font=font)
    if person_views:
        summary = f"persons {len(person_views)} | violations {violations}"
        draw.text((max(5, img_w - 230), 3), summary, fill=(255, 255, 255), font=font)
    return image


def draw_results_cv(
    frame_rgb,
    boxes,
    classes,
    scores,
    latency_ms,
    total_ms=None,
    person_views=None,
    show_compliant=False,
    paper_style=False,
):
    import cv2

    frame_bgr = frame_rgb[:, :, ::-1].copy()
    img_h, img_w = frame_bgr.shape[:2]
    person_views = person_views if person_views is not None else build_person_views(boxes, classes, scores)

    violations = 0
    for person_view in person_views:
        ymin, xmin, ymax, xmax = get_person_display_box(person_view, paper_style=paper_style)
        x1, y1 = int(xmin * img_w), int(ymin * img_h)
        x2, y2 = int(xmax * img_w), int(ymax * img_h)

        if person_view["missing"]:
            violations += 1
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 0, 255), 3, cv2.LINE_AA)
            text = f"VIOLATION {person_view['score']:.0%}"
            (text_w, text_h), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_y = max(text_h + 4, y1 - 6)
            cv2.rectangle(
                frame_bgr,
                (x1, text_y - text_h - baseline - 6),
                (x1 + text_w + 8, text_y),
                (0, 0, 255),
                -1,
            )
            cv2.putText(
                frame_bgr,
                text,
                (x1 + 4, text_y - baseline - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

            if not person_view["has_helmet"]:
                draw_tagged_box_cv(
                    frame_bgr,
                    compute_head_alert_rect(person_view, img_w, img_h, score_threshold=0.2),
                    "NoHelmet",
                    (0, 0, 255),
                )

            if not person_view["has_vest"]:
                draw_tagged_box_cv(
                    frame_bgr,
                    compute_vest_alert_rect(person_view, img_w, img_h, score_threshold=0.2),
                    "NoVest",
                    (0, 0, 255),
                )
        else:
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 160, 0), 3 if paper_style else 2, cv2.LINE_AA)
            if not show_compliant:
                continue
            if paper_style:
                continue
            text = f"OK {person_view['score']:.0%}"
            cv2.putText(
                frame_bgr,
                text,
                (x1 + 4, max(14, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (0, 160, 0),
                1,
                cv2.LINE_AA,
            )

    info_parts = [f"det {latency_ms:.0f}ms {1000 / max(latency_ms, 1e-3):.1f}fps"]
    if total_ms is not None:
        info_parts.append(f"loop {total_ms:.0f}ms {1000 / max(total_ms, 1e-3):.1f}fps")
    info = " | ".join(info_parts)

    cv2.rectangle(frame_bgr, (0, 0), (img_w, 28), (0, 0, 0), -1)
    cv2.putText(
        frame_bgr,
        info,
        (6, 19),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if person_views:
        total_workers = len(person_views)
        if violations > 0:
            summary = f"W:{total_workers} V:{violations}"
            summary_color = (0, 0, 255)
        else:
            summary = f"W:{total_workers} ALL OK"
            summary_color = (0, 200, 0)
        cv2.putText(
            frame_bgr,
            summary,
            (max(6, img_w - 180), 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            summary_color,
            1,
            cv2.LINE_AA,
        )
    return frame_bgr


def draw_pose_cv(frame_bgr, poses, score_threshold):
    import cv2

    img_h, img_w = frame_bgr.shape[:2]
    for pose in poses:
        for (start_idx, end_idx), color in zip(POSE_EDGES, POSE_EDGE_COLORS):
            if pose[start_idx][2] < score_threshold or pose[end_idx][2] < score_threshold:
                continue
            x1 = int(pose[start_idx][1] * img_w)
            y1 = int(pose[start_idx][0] * img_h)
            x2 = int(pose[end_idx][1] * img_w)
            y2 = int(pose[end_idx][0] * img_h)
            color_bgr = (int(color[2]), int(color[1]), int(color[0]))
            cv2.line(frame_bgr, (x1, y1), (x2, y2), color_bgr, 2, cv2.LINE_AA)

        for idx, (kp_y, kp_x, kp_score) in enumerate(pose):
            if kp_score < score_threshold:
                continue
            x = int(kp_x * img_w)
            y = int(kp_y * img_h)
            cv2.circle(frame_bgr, (x, y), 4, (64, 64, 255), -1, cv2.LINE_AA)
            if idx == 0:
                cv2.putText(
                    frame_bgr,
                    "nose",
                    (x + 6, max(14, y - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
    return frame_bgr


class PreviewWindow:
    def __init__(self, window_name="PPE Detection", scale=1.0, max_fps=12.0):
        self.window_name = window_name
        self.scale = float(np.clip(scale, PREVIEW_MIN_SCALE, PREVIEW_MAX_SCALE))
        self.max_fps = max(1.0, float(max_fps))
        self.base_size = None
        self.created = False
        self.available = True
        self.warning_printed = False
        self._lock = threading.Lock()
        self._latest_frame = None
        self._pending_resize = False
        self._quit_requested = threading.Event()
        self._stop_requested = threading.Event()
        self._worker = None

    def disable(self, message):
        if not self.warning_printed:
            print(message)
            self.warning_printed = True
        self.available = False
        self._quit_requested.set()
        self._stop_requested.set()

    def apply_size(self):
        import cv2

        if not self.available or not self.created or self.base_size is None:
            return
        width = max(1, int(self.base_size[0] * self.scale))
        height = max(1, int(self.base_size[1] * self.scale))
        try:
            cv2.resizeWindow(self.window_name, width, height)
        except cv2.error:
            self.disable("Preview disabled: OpenCV GUI backend is unavailable in this environment.")

    def _ensure_worker(self):
        if self._worker is None:
            self._worker = threading.Thread(target=self._ui_loop, daemon=True)
            self._worker.start()

    def show(self, image, assume_bgr=False, copy_frame=True):
        if not self.available:
            return False
        if isinstance(image, Image.Image):
            frame = np.asarray(image)
            frame_bgr = frame[:, :, ::-1].copy()
            original_size = image.size
        else:
            frame = np.asarray(image)
            if frame.ndim != 3 or frame.shape[2] != 3:
                raise ValueError("Preview frame must be an RGB/BGR image.")
            frame_bgr = frame.copy() if assume_bgr and copy_frame else frame
            if not assume_bgr:
                frame_bgr = frame[:, :, ::-1].copy()
            original_size = (frame.shape[1], frame.shape[0])

        with self._lock:
            self._latest_frame = frame_bgr
            if self.base_size != original_size:
                self.base_size = original_size
                self._pending_resize = True

        self._ensure_worker()
        return True

    def handle_key(self, key):
        if not self.available:
            return None
        if key in (ord("q"), 27):
            return "quit"
        if key in (ord("+"), ord("=")):
            self.scale = min(PREVIEW_MAX_SCALE, self.scale + PREVIEW_SCALE_STEP)
            self._pending_resize = True
            print(f"Preview scale: {self.scale:.2f}x")
        elif key == ord("-"):
            self.scale = max(PREVIEW_MIN_SCALE, self.scale - PREVIEW_SCALE_STEP)
            self._pending_resize = True
            print(f"Preview scale: {self.scale:.2f}x")
        elif key == ord("0"):
            self.scale = 1.0
            self._pending_resize = True
            print(f"Preview scale: {self.scale:.2f}x")
        return None

    def should_quit(self):
        return self._quit_requested.is_set()

    def close(self):
        self._stop_requested.set()
        if self._worker and self._worker.is_alive():
            self._worker.join(timeout=1.0)

    def _ui_loop(self):
        import cv2

        frame_interval = 1.0 / self.max_fps
        while not self._stop_requested.is_set():
            loop_started = time.perf_counter()
            frame_bgr = None
            resize_needed = False

            with self._lock:
                if self._latest_frame is not None:
                    frame_bgr = self._latest_frame
                    self._latest_frame = None
                if self._pending_resize:
                    resize_needed = True
                    self._pending_resize = False

            if not self.available:
                break

            if frame_bgr is not None:
                if not self.created:
                    try:
                        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
                    except cv2.error:
                        self.disable("Preview disabled: OpenCV GUI backend is unavailable in this environment.")
                        break
                    self.created = True
                    resize_needed = True
                if resize_needed:
                    self.apply_size()
                try:
                    cv2.imshow(self.window_name, frame_bgr)
                except cv2.error:
                    self.disable("Preview disabled: OpenCV GUI backend is unavailable in this environment.")
                    break

            if self.created:
                try:
                    visible = cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE)
                    if visible < 1:
                        self._quit_requested.set()
                        break
                except cv2.error:
                    pass

            try:
                key = cv2.waitKey(1) & 0xFF
            except cv2.error:
                self.disable("Preview disabled: OpenCV GUI backend is unavailable in this environment.")
                break
            if self.handle_key(key) == "quit":
                self._quit_requested.set()
                break

            elapsed = time.perf_counter() - loop_started
            if elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        if self.created:
            try:
                cv2.destroyWindow(self.window_name)
            except cv2.error:
                pass


def safe_destroy_windows():
    try:
        import cv2

        cv2.destroyAllWindows()
    except Exception:
        pass


def test_image(args):
    preview = PreviewWindow(scale=args.preview_scale, max_fps=args.preview_max_fps) if args.preview else None
    interp, inp, out, size, model_info = load_runtime(args)
    pose_model = (
        load_pose_model(
            args.pose_model,
            num_threads=args.pose_threads,
            delegate=args.pose_delegate,
            edgetpu_device=args.pose_edgetpu_device,
        )
        if args.pose_model
        else None
    )

    image = Image.open(args.image).convert("RGB")
    print(f"Image: {args.image} ({image.size[0]}x{image.size[1]})")

    dummy = Image.new("RGB", (size[1], size[0]))
    run_inference(interp, inp, out, dummy, size, model_info)

    boxes, classes, scores, latency = run_inference(interp, inp, out, image, size, model_info)
    person_views = build_person_views(boxes, classes, scores)
    if pose_model:
        person_views, _ = attach_pose_views(image, person_views, pose_model, args, frame_count=0, cached_pose_views=[])
        person_views = rematch_person_views_with_pose(person_views, boxes, classes, scores)

    print(f"\nLatency: {latency:.1f}ms ({1000/latency:.1f} FPS)")
    print(f"Persons: {len(person_views)}")
    violations = 0
    for idx, person_view in enumerate(person_views, start=1):
        if person_view["missing"]:
            violations += 1
            print(f"  person {idx}: VIOLATION ({', '.join(person_view['missing'])})")
        else:
            print(f"  person {idx}: OK")
    print(f"Violations: {violations}")

    if not args.benchmark:
        result = draw_results(
            image.copy(),
            boxes,
            classes,
            scores,
            latency,
            person_views=person_views,
            paper_style=args.paper_style,
        )
        if args.pose_overlay:
            poses = collect_pose_arrays(person_views)
            if poses:
                result = draw_pose(result, poses, args.pose_score_threshold)
        out_path = Path(args.image).stem + "_result.jpg"
        result.save(out_path, quality=90)
        print(f"Saved: {out_path}")
        if preview:
            shown = preview.show(result)
            if shown:
                print("Preview controls: '+' zoom in, '-' zoom out, '0' reset, 'q' close.")
                try:
                    while not preview.should_quit():
                        time.sleep(0.05)
                finally:
                    preview.close()
                    safe_destroy_windows()

    if args.benchmark:
        print("\nBenchmark (10 runs):")
        times = []
        for _ in range(10):
            _, _, _, t = run_inference(interp, inp, out, image, size, model_info)
            times.append(t)
        avg = np.mean(times)
        std = np.std(times)
        print(f"  Avg: {avg:.1f}ms +/- {std:.1f}ms")
        print(f"  FPS: {1000/avg:.1f}")
        print(f"  Min: {min(times):.1f}ms  Max: {max(times):.1f}ms")

    close_runtime(interp, model_info)


def list_available_cameras():
    available = {"picamera2": [], "opencv": []}

    try:
        from picamera2 import Picamera2

        available["picamera2"] = Picamera2.global_camera_info()
    except Exception:
        available["picamera2"] = []

    available["opencv"] = sorted(glob.glob("/dev/video*"))
    return available


def print_available_cameras():
    available = list_available_cameras()
    if available["picamera2"]:
        print("Picamera2 cameras:")
        for idx, info in enumerate(available["picamera2"]):
            print(f"  [{idx}] {info}")
    else:
        print("Picamera2 cameras: none")

    if available["opencv"]:
        print("OpenCV video devices:")
        for dev in available["opencv"]:
            print(f"  {dev}")
    else:
        print("OpenCV video devices: none")


class OpenCVLatestFrameCamera:
    def __init__(self, capture, read_timeout=1.0, read_mode="latest"):
        self.capture = capture
        self.read_timeout = float(read_timeout)
        self.read_mode = str(read_mode or "latest")
        self._lock = threading.Lock()
        self._ready = threading.Event()
        self._stopped = threading.Event()
        self._latest_frame = None
        self._latest_frame_id = 0
        self._reader = threading.Thread(target=self._reader_loop, daemon=True)
        self._reader.start()
        if not self._ready.wait(timeout=self.read_timeout):
            self.release()
            raise RuntimeError("Timed out waiting for the USB camera to deliver the first frame.")

    def _reader_loop(self):
        while not self._stopped.is_set():
            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.005)
                continue
            frame_rgb = frame[:, :, ::-1].copy()
            with self._lock:
                self._latest_frame = frame_rgb
                self._latest_frame_id += 1
            self._ready.set()

    def read_rgb(self, last_frame_id=None, timeout=None, require_new_frame=False, copy_frame=False):
        wait_timeout = self.read_timeout if timeout is None else max(0.0, float(timeout))
        deadline = time.perf_counter() + wait_timeout

        while True:
            remaining = deadline - time.perf_counter()
            if not self._ready.wait(timeout=max(0.0, remaining)):
                raise RuntimeError("Timed out waiting for a fresh frame from the USB camera.")

            with self._lock:
                latest = self._latest_frame
                frame_id = self._latest_frame_id

            if latest is None:
                if time.perf_counter() >= deadline:
                    raise RuntimeError("USB camera worker has not produced a frame yet.")
                time.sleep(0.002)
                continue

            if not require_new_frame or last_frame_id is None or frame_id != last_frame_id:
                return (latest.copy() if copy_frame else latest), frame_id

            if time.perf_counter() >= deadline:
                return (latest.copy() if copy_frame else latest), frame_id

            time.sleep(0.002)

    def release(self):
        self._stopped.set()
        try:
            self.capture.release()
        except Exception:
            pass
        self._reader.join(timeout=1.0)


def get_opencv_capture(camera):
    return camera.capture if isinstance(camera, OpenCVLatestFrameCamera) else camera


def decode_fourcc(value):
    int_value = int(value or 0)
    chars = [chr((int_value >> (8 * idx)) & 0xFF) for idx in range(4)]
    return "".join(chars).strip("\x00") or "n/a"


def configure_opencv_capture(cap, args):
    import cv2

    if args.camera_fourcc:
        fourcc = str(args.camera_fourcc).strip().upper()
        if len(fourcc) == 4:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))

    if args.camera_buffer_size >= 0 and hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
        cap.set(cv2.CAP_PROP_BUFFERSIZE, int(args.camera_buffer_size))

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.camera_width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.camera_height))
    if args.camera_fps > 0:
        cap.set(cv2.CAP_PROP_FPS, float(args.camera_fps))


def describe_camera(backend, camera):
    if backend != "opencv":
        return None

    cap = get_opencv_capture(camera)
    try:
        width = int(cap.get(3) or 0)
        height = int(cap.get(4) or 0)
        fps = float(cap.get(5) or 0.0)
        fourcc = decode_fourcc(cap.get(6))
        mode = getattr(camera, "read_mode", "latest") if isinstance(camera, OpenCVLatestFrameCamera) else "blocking"
        return f"OpenCV config: {width}x{height} @ {fps:.1f} FPS, FOURCC={fourcc}, read_mode={mode}"
    except Exception:
        return None


def open_opencv_camera(args, allow_missing=False):
    try:
        import cv2
    except ImportError:
        if allow_missing:
            return None
        raise RuntimeError("OpenCV not installed. Run: sudo apt install python3-opencv")

    cap = cv2.VideoCapture(args.camera_index, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap.release()
        if allow_missing:
            return None
        raise RuntimeError(f"Could not open USB camera at index {args.camera_index}")

    configure_opencv_capture(cap, args)
    if args.camera_read_mode in {"latest", "latest_fresh"}:
        read_timeout = max(1.0, 2.0 / max(1.0, float(getattr(args, "camera_fps", 30) or 30)))
        return OpenCVLatestFrameCamera(cap, read_timeout=read_timeout, read_mode=args.camera_read_mode)
    return cap


def open_camera(args):
    backend = args.camera_backend.lower()

    if backend in ("auto", "picamera2"):
        try:
            from picamera2 import Picamera2

            cam_w = int(args.camera_width)
            cam_h = int(args.camera_height)
            info = Picamera2.global_camera_info()
            if info:
                cam = Picamera2()
                cam.configure(
                    cam.create_preview_configuration(main={"size": (cam_w, cam_h), "format": "RGB888"})
                )
                cam.start()
                return "picamera2", cam
            if backend == "picamera2":
                raise RuntimeError("Picamera2 is installed, but no CSI camera is available.")
        except ImportError:
            if backend == "picamera2":
                raise RuntimeError("picamera2 not installed. Run: sudo apt install python3-picamera2")

    if backend == "auto":
        camera = open_opencv_camera(args, allow_missing=True)
        if camera is not None:
            return "opencv", camera

    if backend == "opencv":
        return "opencv", open_opencv_camera(args, allow_missing=False)

    raise RuntimeError("No usable camera backend found. Try --list-cameras first.")


def capture_frame(backend, camera, args=None, last_frame_id=None):
    if backend == "picamera2":
        return camera.capture_array(), None

    if backend == "opencv":
        if isinstance(camera, OpenCVLatestFrameCamera):
            require_new_frame = getattr(camera, "read_mode", "latest") == "latest_fresh"
            return camera.read_rgb(last_frame_id=last_frame_id, require_new_frame=require_new_frame, copy_frame=False)

        flush_grabs = max(0, int(getattr(args, "camera_flush_grabs", 0) or 0))
        for _ in range(flush_grabs):
            camera.grab()
        ok, frame = camera.read()
        if not ok:
            raise RuntimeError("Failed to read frame from USB camera.")
        return frame[:, :, ::-1], None

    raise RuntimeError(f"Unsupported camera backend: {backend}")


def capture_frame_latest(backend, camera, args=None, last_frame_id=None):
    return capture_frame(backend, camera, args=args, last_frame_id=last_frame_id)


def close_camera(backend, camera):
    if backend == "picamera2":
        camera.stop()
    elif backend == "opencv":
        if isinstance(camera, OpenCVLatestFrameCamera):
            camera.release()
        else:
            camera.release()


def test_camera(args):
    preview = PreviewWindow(scale=args.preview_scale, max_fps=args.preview_max_fps) if args.preview else None
    interp, inp, out, size, model_info = load_runtime(args)
    person_runtime = load_optional_person_runtime(args)
    pose_model = (
        load_pose_model(
            args.pose_model,
            num_threads=args.pose_threads,
            delegate=args.pose_delegate,
            edgetpu_device=args.pose_edgetpu_device,
        )
        if args.pose_model
        else None
    )
    backend, camera = open_camera(args)
    print(f"Camera backend: {backend}")
    camera_summary = describe_camera(backend, camera)
    if camera_summary:
        print(camera_summary)
    if preview:
        print("Camera started. Preview controls: '+' zoom in, '-' zoom out, '0' reset, 'q' stop.\n")
    else:
        print("Camera started. Press Ctrl+C to stop.\n")

    try:
        frame_count = 0
        infer_times = []
        loop_times = []
        cached_pose_views = []
        recent_person_views = []
        last_camera_frame_id = None
        while True:
            if args.camera_max_frames and frame_count >= args.camera_max_frames:
                break

            t_loop_start = time.perf_counter()
            frame, last_camera_frame_id = capture_frame_latest(
                backend,
                camera,
                args=args,
                last_frame_id=last_camera_frame_id,
            )
            boxes, classes, scores, latency = run_inference(
                interp, inp, out, frame, size, model_info
            )
            merged_boxes, merged_classes, merged_scores = boxes, classes, scores
            if person_runtime is not None:
                p_interp, p_inp, p_out, p_size, p_model_info = person_runtime["runtime"]
                person_boxes, person_classes, person_scores, person_latency = run_inference(
                    p_interp,
                    p_inp,
                    p_out,
                    frame,
                    p_size,
                    p_model_info,
                )
                latency += person_latency
                merged_boxes, merged_classes, merged_scores, person_views = build_person_views_hybrid(
                    boxes,
                    classes,
                    scores,
                    person_boxes=person_boxes,
                    person_classes=person_classes,
                    person_scores=person_scores,
                    person_labels_map=person_runtime["labels_map"],
                    person_threshold=person_runtime["threshold"],
                )
            else:
                person_views = build_person_views(boxes, classes, scores)
            person_views, cached_pose_views = attach_pose_views(
                frame,
                person_views,
                pose_model,
                args,
                frame_count,
                cached_pose_views,
            )
            if pose_model:
                person_views = rematch_person_views_with_pose(
                    person_views,
                    merged_boxes,
                    merged_classes,
                    merged_scores,
                )
            person_views, recent_person_views = stabilize_person_views(
                person_views,
                recent_person_views,
                frame_count,
            )
            if preview:
                should_draw = (frame_count % max(1, args.preview_every)) == 0
                if should_draw:
                    loop_ms_so_far = (time.perf_counter() - t_loop_start) * 1000
                    frame_bgr = draw_results_cv(
                        frame,
                        merged_boxes,
                        merged_classes,
                        merged_scores,
                        latency,
                        total_ms=loop_ms_so_far,
                        person_views=person_views,
                        show_compliant=args.show_compliant,
                        paper_style=args.paper_style,
                    )
                    if args.pose_overlay:
                        poses = collect_pose_arrays(person_views)
                        if poses:
                            frame_bgr = draw_pose_cv(frame_bgr, poses, args.pose_score_threshold)
                    shown = preview.show(frame_bgr, assume_bgr=True, copy_frame=False)
                    if not shown:
                        preview = None
                if preview and preview.should_quit():
                    break

            frame_count += 1
            infer_times.append(latency)
            loop_times.append((time.perf_counter() - t_loop_start) * 1000)

            if frame_count % 10 == 0:
                avg_infer = np.mean(infer_times[-10:])
                avg_loop = np.mean(loop_times[-10:])
                violations = sum(1 for person_view in person_views if person_view["missing"])
                print(
                    f"Frame {frame_count}: infer {avg_infer:.0f}ms ({1000/avg_infer:.1f} FPS) "
                    f"| loop {avg_loop:.0f}ms ({1000/avg_loop:.1f} FPS) | persons {len(person_views)} | violations {violations}"
                )

    except KeyboardInterrupt:
        print(f"\nStopped after {frame_count} frames")
        if infer_times:
            avg_infer = np.mean(infer_times)
            avg_loop = np.mean(loop_times)
            print(f"Average infer: {avg_infer:.1f}ms ({1000/avg_infer:.1f} FPS)")
            print(f"Average loop: {avg_loop:.1f}ms ({1000/avg_loop:.1f} FPS)")
    finally:
        close_camera(backend, camera)
        close_runtime(interp, model_info)
        if person_runtime is not None:
            close_runtime(person_runtime["runtime"][0], person_runtime["runtime"][4])
        if preview:
            preview.close()
        if args.preview:
            safe_destroy_windows()


def test_video(args):
    import cv2

    preview = PreviewWindow(scale=args.preview_scale, max_fps=args.preview_max_fps) if args.preview else None
    interp, inp, out, size, model_info = load_runtime(args)
    pose_model = (
        load_pose_model(
            args.pose_model,
            num_threads=args.pose_threads,
            delegate=args.pose_delegate,
            edgetpu_device=args.pose_edgetpu_device,
        )
        if args.pose_model
        else None
    )

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {args.video}")

    src_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if src_fps <= 0:
        src_fps = 15.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    output_path = None
    writer = None
    if args.save_video or args.video_out:
        output_path = resolve_output_video_path(args.video, args.video_out)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(output_path, fourcc, src_fps, (frame_w, frame_h))
        if not writer.isOpened():
            writer.release()
            writer = None
            raise RuntimeError(f"Could not open video writer: {output_path}")

    print(f"Video: {args.video}")
    print(f"  Size: {frame_w}x{frame_h}")
    print(f"  FPS: {src_fps:.2f}")
    if total_frames > 0:
        print(f"  Frames: {total_frames}")
    if output_path:
        print(f"  Saving: {output_path}")
    if preview:
        print("Video started. Preview controls: '+' zoom in, '-' zoom out, '0' reset, 'q' stop.\n")

    frame_count = 0
    infer_times = []
    loop_times = []
    total_persons = 0
    total_violations = 0
    missing_helmet_count = 0
    missing_vest_count = 0
    cached_pose_views = []
    recent_person_views = []

    try:
        while True:
            if args.video_max_frames and frame_count >= args.video_max_frames:
                break

            t_loop_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                break

            frame_rgb = frame_bgr[:, :, ::-1]
            boxes, classes, scores, latency = run_inference(
                interp, inp, out, frame_rgb, size, model_info
            )
            person_views = build_person_views(boxes, classes, scores)
            person_views, cached_pose_views = attach_pose_views(
                frame_rgb,
                person_views,
                pose_model,
                args,
                frame_count,
                cached_pose_views,
            )
            if pose_model:
                person_views = rematch_person_views_with_pose(person_views, boxes, classes, scores)
            person_views, recent_person_views = stabilize_person_views(
                person_views,
                recent_person_views,
                frame_count,
            )
            total_persons += len(person_views)
            frame_violations = sum(1 for person_view in person_views if person_view["missing"])
            total_violations += frame_violations
            missing_helmet_count += sum(1 for person_view in person_views if not person_view["has_helmet"])
            missing_vest_count += sum(1 for person_view in person_views if not person_view["has_vest"])

            need_render = bool(preview or writer)
            if need_render:
                loop_ms_so_far = (time.perf_counter() - t_loop_start) * 1000
                render_bgr = draw_results_cv(
                    frame_rgb,
                    boxes,
                    classes,
                    scores,
                    latency,
                    total_ms=loop_ms_so_far,
                    person_views=person_views,
                    show_compliant=args.show_compliant,
                    paper_style=args.paper_style,
                )
                if args.pose_overlay:
                    poses = collect_pose_arrays(person_views)
                    if poses:
                        render_bgr = draw_pose_cv(render_bgr, poses, args.pose_score_threshold)
                if writer:
                    writer.write(render_bgr)
                if preview:
                    should_draw = (frame_count % max(1, args.preview_every)) == 0
                    if should_draw:
                        shown = preview.show(render_bgr, assume_bgr=True, copy_frame=False)
                        if not shown:
                            preview = None
                    if preview and preview.should_quit():
                        break

            frame_count += 1
            infer_times.append(latency)
            loop_times.append((time.perf_counter() - t_loop_start) * 1000)

            if frame_count % 30 == 0:
                avg_infer = np.mean(infer_times[-30:])
                avg_loop = np.mean(loop_times[-30:])
                print(
                    f"Frame {frame_count}: infer {avg_infer:.0f}ms ({1000/avg_infer:.1f} FPS) "
                    f"| loop {avg_loop:.0f}ms ({1000/avg_loop:.1f} FPS) "
                    f"| persons {total_persons} | violations {total_violations}"
                )

    finally:
        cap.release()
        if writer:
            writer.release()
        close_runtime(interp, model_info)
        if preview:
            preview.close()
        if args.preview:
            safe_destroy_windows()

    print(f"\nProcessed frames: {frame_count}")
    if infer_times:
        avg_infer = float(np.mean(infer_times))
        avg_loop = float(np.mean(loop_times))
        print(f"Average infer: {avg_infer:.1f}ms ({1000/avg_infer:.1f} FPS)")
        print(f"Average loop: {avg_loop:.1f}ms ({1000/avg_loop:.1f} FPS)")
    print(f"Total person views: {total_persons}")
    print(f"Total violations: {total_violations}")
    print(f"  missing helmet: {missing_helmet_count}")
    print(f"  missing vest: {missing_vest_count}")
    if output_path:
        print(f"Saved video: {output_path}")


def main():
    p = argparse.ArgumentParser(description="PPE Detection on RPi5")
    p.add_argument("--model", default=MODEL_PATH, help="TFLite model path")
    p.add_argument("--image", help="Test image path")
    p.add_argument("--video", help="Test video path")
    p.add_argument("--camera", action="store_true", help="Run live camera inference")
    p.add_argument(
        "--camera-backend",
        default="auto",
        choices=["auto", "picamera2", "opencv"],
        help="Camera backend for --camera",
    )
    p.add_argument("--camera-index", type=int, default=0, help="OpenCV camera index")
    p.add_argument("--camera-width", type=int, default=640, help="Requested camera width")
    p.add_argument("--camera-height", type=int, default=480, help="Requested camera height")
    p.add_argument("--camera-fps", type=int, default=30, help="Requested OpenCV camera FPS")
    p.add_argument("--camera-buffer-size", type=int, default=1, help="Requested OpenCV capture buffer size")
    p.add_argument(
        "--camera-fourcc",
        default="MJPG",
        help="Preferred OpenCV camera codec, e.g. MJPG or YUYV",
    )
    p.add_argument(
        "--camera-read-mode",
        default="latest",
        choices=["latest", "latest_fresh", "blocking"],
        help="Use the latest available frame, wait for a fresh frame, or use a blocking read loop for OpenCV cameras",
    )
    p.add_argument(
        "--camera-flush-grabs",
        type=int,
        default=2,
        help="Extra OpenCV grab() calls before read when using blocking mode",
    )
    p.add_argument("--camera-max-frames", type=int, default=0, help="Stop after this many frames when using --camera")
    p.add_argument(
        "--delegate",
        default="cpu",
        choices=["cpu", "edgetpu", "edgetpu_dual"],
        help="Inference delegate",
    )
    p.add_argument(
        "--edgetpu-device",
        default=DEFAULT_EDGETPU_DEVICE,
        help="Edge TPU device for the main PPE model (for example: auto-pci, pci:0, pci:1, usb:0)",
    )
    p.add_argument("--dual-model-seg0", default=DUAL_MODEL_SEG0_PATH, help="Segment 0 model path for dual Edge TPU mode")
    p.add_argument("--dual-model-seg1", default=DUAL_MODEL_SEG1_PATH, help="Segment 1 model path for dual Edge TPU mode")
    p.add_argument("--dual-device0", default="pci:0", help="Device for segment 0 in dual Edge TPU mode")
    p.add_argument("--dual-device1", default="usb:0", help="Device for segment 1 in dual Edge TPU mode")
    p.add_argument("--threads", type=int, default=4, help="CPU threads")
    p.add_argument("--benchmark", action="store_true", help="Run benchmark (10 iterations)")
    p.add_argument("--save-video", action="store_true", help="Save annotated output when using --video")
    p.add_argument("--video-out", help="Optional path for the annotated output video")
    p.add_argument("--video-max-frames", type=int, default=0, help="Stop after this many frames when using --video")
    p.add_argument("--list-cameras", action="store_true", help="List available camera backends and devices")
    p.add_argument("--preview", action="store_true", help="Show a live preview window with drawn detections")
    p.add_argument(
        "--preview-scale",
        type=float,
        default=1.0,
        help="Rendered preview scale (resizable later with + / - / 0)",
    )
    p.add_argument(
        "--preview-every",
        type=int,
        default=1,
        help="Draw and show every Nth frame while still running inference on all frames",
    )
    p.add_argument(
        "--preview-max-fps",
        type=float,
        default=12.0,
        help="Maximum preview window refresh rate (useful over XRDP)",
    )
    p.add_argument("--show-compliant", action="store_true", help="Also draw compliant workers in the preview/output")
    p.add_argument(
        "--paper-style",
        action="store_true",
        help="Use large full-person boxes plus pose-guided NoHelmet/NoVest boxes like the paper-style result images",
    )
    p.add_argument(
        "--detector-resize-mode",
        default="letterbox",
        choices=["letterbox", "stretch"],
        help="Resize strategy before detector inference. letterbox keeps aspect ratio and restores boxes to the original frame.",
    )
    p.add_argument(
        "--label-preset",
        default="auto",
        choices=["auto", "default", "final_ft_b8"],
        help="Class-order preset. final_ft_b8 means 0=person, 1=vest, 2=helmet.",
    )
    p.add_argument("--person-model", help="Optional secondary detector used only for person boxes")
    p.add_argument(
        "--person-model-label-preset",
        default="default",
        choices=["auto", "default", "final_ft_b8"],
        help="Class-order preset for --person-model",
    )
    p.add_argument(
        "--person-model-threshold",
        type=float,
        default=0.30,
        help="Minimum score threshold for person boxes coming from --person-model",
    )
    p.add_argument(
        "--person-model-resize-mode",
        default="letterbox",
        choices=["letterbox", "stretch"],
        help="Resize strategy for the optional --person-model",
    )
    p.add_argument("--person-threshold", type=float, help="Override person detection score threshold")
    p.add_argument("--helmet-threshold", type=float, help="Override helmet detection score threshold")
    p.add_argument("--vest-threshold", type=float, help="Override vest detection score threshold")
    p.add_argument("--pose-model", help="Optional MoveNet TFLite model path for pose overlay")
    p.add_argument("--pose-overlay", action="store_true", help="Draw the pose skeleton when using --pose-model")
    p.add_argument(
        "--pose-delegate",
        default="cpu",
        choices=["cpu", "edgetpu"],
        help="Inference delegate for the optional pose model",
    )
    p.add_argument(
        "--pose-edgetpu-device",
        default=DEFAULT_EDGETPU_DEVICE,
        help="Edge TPU device for the optional pose model when --pose-delegate edgetpu",
    )
    p.add_argument("--pose-threads", type=int, default=2, help="CPU threads for the optional pose model")
    p.add_argument(
        "--pose-score-threshold",
        type=float,
        default=0.2,
        help="Minimum keypoint confidence to draw",
    )
    p.add_argument(
        "--pose-person-threshold",
        type=float,
        default=0.28,
        help="Minimum person detection confidence before running MoveNet on that box",
    )
    p.add_argument(
        "--pose-max-persons",
        type=int,
        default=20,
        help="Maximum number of person boxes to run MoveNet on",
    )
    p.add_argument(
        "--pose-every",
        type=int,
        default=2,
        help="Run MoveNet every N frames and reuse the last pose estimate in between",
    )
    p.add_argument(
        "--pose-all-persons",
        action="store_true",
        help="Run MoveNet on all qualifying person boxes instead of only missing-PPE persons",
    )
    p.add_argument(
        "--pose-crop-scale",
        type=float,
        default=1.25,
        help="How much to expand the detected person box before pose inference",
    )
    args = p.parse_args()
    args.model = resolve_input_path(args.model)
    if args.image:
        args.image = resolve_input_path(args.image)
    if args.video:
        args.video = resolve_input_path(args.video)
    if args.pose_model:
        args.pose_model = resolve_input_path(args.pose_model)
    if args.person_model:
        args.person_model = resolve_input_path(args.person_model)
    configure_label_preset(
        args.label_preset,
        args.model if args.delegate != "edgetpu_dual" else args.dual_model_seg0,
    )
    apply_conf_threshold_overrides(args)

    if args.list_cameras:
        print_available_cameras()
    elif args.camera:
        test_camera(args)
    elif args.video:
        test_video(args)
    elif args.image:
        test_image(args)
    else:
        p.print_help()
        print("\nQuick start:")
        print("  python3 test_ppe_rpi5.py --image test.jpg")
        print("  python3 test_ppe_rpi5.py --image test.jpg --benchmark")
        print("  python3 test_ppe_rpi5.py --video test.mp4 --save-video")
        print("  python3 test_ppe_rpi5.py --camera")
        print("  python3 test_ppe_rpi5.py --camera --camera-backend opencv")
        print("  python3 test_ppe_rpi5.py --image test.jpg --delegate edgetpu --edgetpu-device auto-pci")
        print("  python3 test_ppe_rpi5.py --camera --preview --pose-model movenet_lightning_int8.tflite")


if __name__ == "__main__":
    main()
