from __future__ import annotations

import json
import tempfile
import threading
import time
from collections import deque
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import WebRtcMode, VideoProcessorBase, webrtc_streamer

import models  # triggers registration
from core.types import DetectionResult, FrameResult, NavigationTimeline
from core.registry import create, get_class, list_models
from core.video import export_annotated_video, get_video_info, _reencode_h264
from models.vlm.lfm25 import GENERIC_JSON_SCHEMA_PROMPT, NAV_STATE_PROMPT
from pipeline.evaluator import evaluate_nav_accuracy, parse_manual_labels
from pipeline.exporter import export_ground_truth, export_ground_truth_json
from pipeline.interactions import InteractionConfig, InteractionEngine
from pipeline.nav_state import build_nav_timeline
from pipeline.runner import Pipeline
from pipeline.visualizer import (
    draw_all,
    draw_detections,
    draw_hands_and_interactions,
    draw_segmentation,
    draw_temporal_changes,
    draw_tracking,
)


st.set_page_config(layout="wide", page_title="World2Data - Live")
st.title("Live Perception")


def _build_analysis_results_json(
    results: list[FrameResult],
    info: dict,
    nav_timeline: NavigationTimeline | None,
) -> str:
    payload = export_ground_truth(results, info, nav_timeline=nav_timeline)
    timing_keys = sorted({k for r in results for k in r.timings.keys()})
    payload["timing_seconds"] = {
        key: [float(r.timings[key]) for r in results if key in r.timings]
        for key in timing_keys
    }
    payload["frames_diagnostics"] = [
        {
            "frame_idx": r.frame_idx,
            "timings": {k: float(v) for k, v in r.timings.items()},
            "num_hands": len(getattr(r, "hand_poses", []) or []),
            "num_interactions": len(getattr(r, "interactions", []) or []),
        }
        for r in results
    ]
    return json.dumps(payload, indent=2, default=str)


def _init_state() -> None:
    defaults = {
        "live_tracker": None,
        "live_segmenter": None,
        "live_detector": None,
        "live_vlm": None,
        "live_state_classifier": None,
        "live_trk_name": "bytetrack",
        "live_seg_name": "rf-detr-seg",
        "live_det_name": "yolov11",
        "live_vlm_name": "gemini3",
        "live_state_classifier_name": "siglip",
        "live_loaded": False,
        "live_auto_loaded": False,
        "live_interaction_engine": None,
        "live_recording": False,
        "live_recording_path": None,
        "live_recorded_video": None,
        "live_results": [],
        "live_nav_timeline": None,
        "live_video_info": None,
        "live_exported_video": None,
        "live_prev_playing": False,
        "live_last_analyzed_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def _pick_model_size(category: str, name: str, label: str):
    cls = get_class(category, name)
    kwargs = {}
    if hasattr(cls, "VALID_SIZES"):
        sizes = list(cls.VALID_SIZES)
        kwargs["model_size"] = st.selectbox(label, sizes, index=0)
    return kwargs


def _safe_unload(obj):
    if obj is not None:
        obj.unload()


def _load_selected_models(
    *,
    enable_tracker: bool,
    enable_seg: bool,
    enable_det: bool,
    enable_vlm: bool,
    enable_state_classifier: bool,
    trk_choice: str,
    seg_choice: str,
    det_choice: str,
    vlm_choice: str,
    state_classifier_choice: str,
    trk_kwargs: dict,
    seg_kwargs: dict,
    det_kwargs: dict,
) -> None:
    _safe_unload(st.session_state["live_tracker"])
    _safe_unload(st.session_state["live_segmenter"])
    _safe_unload(st.session_state["live_detector"])
    _safe_unload(st.session_state["live_vlm"])
    _safe_unload(st.session_state["live_state_classifier"])

    st.session_state["live_tracker"] = None
    st.session_state["live_segmenter"] = None
    st.session_state["live_detector"] = None
    st.session_state["live_vlm"] = None
    st.session_state["live_state_classifier"] = None

    if enable_tracker:
        trk = create("tracker", trk_choice, **trk_kwargs)
        trk.load()
        st.session_state["live_tracker"] = trk
        st.session_state["live_trk_name"] = trk_choice

    if enable_seg:
        seg = create("segmenter", seg_choice, **seg_kwargs)
        seg.load()
        st.session_state["live_segmenter"] = seg
        st.session_state["live_seg_name"] = seg_choice

    if enable_det:
        det = create("detector", det_choice, **det_kwargs)
        det.load()
        st.session_state["live_detector"] = det
        st.session_state["live_det_name"] = det_choice

    if enable_vlm:
        vlm = create("vlm", vlm_choice)
        vlm.load()
        st.session_state["live_vlm"] = vlm
        st.session_state["live_vlm_name"] = vlm_choice

    if enable_state_classifier:
        sc = create("state_classifier", state_classifier_choice)
        sc.load()
        st.session_state["live_state_classifier"] = sc
        st.session_state["live_state_classifier_name"] = state_classifier_choice

    st.session_state["live_loaded"] = any(
        m is not None
        for m in (
            st.session_state["live_tracker"],
            st.session_state["live_segmenter"],
            st.session_state["live_detector"],
            st.session_state["live_vlm"],
            st.session_state["live_state_classifier"],
        )
    )


def _resize_max_side(frame, max_side: int):
    if max_side <= 0:
        return frame
    h, w = frame.shape[:2]
    side = max(h, w)
    if side <= max_side:
        return frame
    scale = float(max_side) / float(side)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_AREA)


def _overlay_vlm_summary(frame, parsed: dict | None, raw_text: str, latency_s: float | None):
    out = frame
    h, w = out.shape[:2]
    panel_w = min(520, int(w * 0.45))
    x0 = w - panel_w - 10
    y0 = 10
    y1 = min(h - 10, 300)
    x1 = w - 10
    # Draw a light, semi-transparent panel to keep the live feed visible.
    overlay = out.copy()
    cv2.rectangle(overlay, (x0, y0), (x1, y1), (245, 245, 245), -1)
    cv2.addWeighted(overlay, 0.35, out, 0.65, 0.0, out)
    cv2.rectangle(out, (x0, y0), (x1, y1), (170, 170, 170), 1)

    lines = []
    header = "LFM state/action"
    if latency_s is not None:
        header += f" ({latency_s * 1000:.0f} ms)"
    lines.append(header)

    added_content = False
    if parsed and isinstance(parsed, dict):
        objects = parsed.get("objects", [])
        if isinstance(objects, list):
            for obj in objects[:6]:
                if not isinstance(obj, dict):
                    continue
                name = str(obj.get("name", "object"))
                state = str(obj.get("state", "unknown"))
                actions = obj.get("robot_actions", [])
                if isinstance(actions, list):
                    act = ", ".join(str(a) for a in actions[:3])
                else:
                    act = str(actions)
                lines.append(f"- {name}: {state}")
                if act:
                    lines.append(f"  actions: {act}")
                added_content = True

        # If parsed exists but doesn't contain object rows, still show something useful.
        if not added_content:
            if "error" in parsed:
                lines.append(f"error: {parsed.get('error')}")
            elif "message" in parsed:
                lines.append(str(parsed.get("message"))[:180])

    if not added_content and raw_text:
        lines.append(raw_text[:180])

    y = y0 + 22
    for line in lines[:12]:
        cv2.putText(out, line[:70], (x0 + 8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1)
        y += 18

    return out


def _overlay_live_task_timeline(
    frame: np.ndarray,
    *,
    action_text: str | None,
    confidence: float | None,
    state_history: deque[str],
) -> np.ndarray:
    out = frame.copy()
    h, w = out.shape[:2]

    # Top task/action banner.
    top_h = 48
    top_overlay = out.copy()
    cv2.rectangle(top_overlay, (0, 0), (w, top_h), (165, 96, 133), -1)
    cv2.addWeighted(top_overlay, 0.45, out, 0.55, 0.0, out)
    cv2.rectangle(out, (0, 0), (w, top_h), (190, 150, 175), 1)

    label = action_text.strip() if action_text else "ANALYZE scene"
    cv2.putText(out, label[:64], (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (245, 245, 245), 2)
    if confidence is not None:
        ctxt = f"{int(max(0.0, min(1.0, confidence)) * 100)}%"
        cv2.putText(out, ctxt, (w - 90, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (245, 245, 245), 2)

    # Bottom state timeline strip.
    strip_h = 20
    y0 = max(0, h - strip_h)
    cv2.rectangle(out, (0, y0), (w, h), (15, 15, 15), -1)
    if state_history:
        state_colors = {
            "open": (80, 200, 120),
            "closed": (90, 90, 230),
            "ajar": (70, 180, 240),
            "blocked": (180, 90, 170),
            "unknown": (140, 140, 140),
        }
        n = len(state_history)
        seg_w = max(1, w // n)
        for i, s in enumerate(state_history):
            color = state_colors.get(str(s).lower(), state_colors["unknown"])
            x0 = i * seg_w
            x1 = w if i == n - 1 else (i + 1) * seg_w
            cv2.rectangle(out, (x0, y0 + 2), (x1, h - 2), color, -1)

    return out


def _build_live_prompt(classes_seen: list[str]) -> str:
    classes = ", ".join(sorted(set(classes_seen))[:20]) if classes_seen else "unknown"
    return (
        "Output ONLY valid JSON with this schema: "
        '{"objects":[{"name":"str","state":"str","robot_actions":["str"]}]}. '
        "Focus on door state (open/closed/ajar/locked/unknown) if a door exists. "
        "For each visible object, provide concise robot actions (e.g. pickup, place, open, close, push, pull, sit, switch_on, switch_off). "
        f"Detected classes hint: {classes}."
    )


def _action_from_object_state(name: str, state: str | None = None) -> str:
    obj = (name or "object").strip().lower()
    obj_norm = obj.replace("_", " ").replace("-", " ").strip()
    obj_token = obj_norm.replace(" ", "")
    stv = (state or "").strip().lower()
    drink_classes = {"cup", "mug", "bottle", "glass", "wine glass", "water bottle"}
    pickup_classes = {
        "can",
        "bowl",
        "plate",
        "box",
        "book",
        "phone",
        "remote",
    }
    openable_classes = {
        "door",
        "cabinet",
        "drawer",
        "refrigerator",
        "microwave",
        "oven",
        "toaster",
    }
    if (
        obj in drink_classes
        or obj_norm in drink_classes
        or "cup" in obj_norm
        or "mug" in obj_norm
        or "bottle" in obj_norm
        or obj_token.endswith("glass")
        or obj_norm == "glass"
    ):
        if stv in {"drinking_pepsi"}:
            return "DRINK PEPSI"
        if stv in {"drinking"}:
            return "DRINK WATER"
        if stv in {"holding"}:
            return "TAKE BOTTLE"
        if stv in {"idle", "unknown", ""}:
            return "IDLE"
        return "DRINK WATER"
    if obj in pickup_classes or "phone" in obj_norm:
        if "phone" in obj_norm and stv in {"using_phone"}:
            return "USE PHONE"
        if "phone" in obj_norm and stv in {"holding"}:
            return "TAKE PHONE"
        if stv in {"idle", "unknown", ""}:
            return "IDLE"
        return f"TAKE {obj_norm.upper()}"
    if obj in openable_classes:
        if stv in {"closed", "blocked", "locked"}:
            return f"OPEN {obj.upper()}"
        if stv in {"open", "ajar"}:
            return f"IDLE {obj.upper()}"
        return f"OPEN {obj.upper()}"
    if obj_norm in {"person", "scene", ""}:
        if stv == "drinking_pepsi":
            return "DRINK PEPSI"
        if stv == "wave":
            return "WAVE"
        if stv == "point":
            return "POINT"
        if stv == "thumbs_up":
            return "THUMBS UP"
        if stv == "drinking":
            return "DRINK WATER"
        if stv == "using_phone":
            return "USE PHONE"
        return "IDLE"
    return "IDLE"


def _is_state_target_class(name: str) -> bool:
    n = (name or "").strip().lower().replace("_", " ").replace("-", " ")
    if not n:
        return False
    direct = {
        "door",
        "cabinet",
        "drawer",
        "refrigerator",
        "microwave",
        "oven",
        "toaster",
        "cup",
        "mug",
        "bottle",
        "water bottle",
        "glass",
        "wine glass",
        "phone",
        "cell phone",
        "person",
    }
    return (
        n in direct
        or "phone" in n
        or "bottle" in n
        or "cup" in n
        or "mug" in n
        or n.endswith("glass")
    )


def _pick_primary_action_class(action_boxes: list, frame_shape: tuple[int, int]) -> str | None:
    """Pick a likely actionable object class for live action text.

    Uses class priority + confidence + box prominence and penalizes off-center boxes.
    """
    if not action_boxes:
        return None
    h, w = frame_shape[:2]
    if h <= 0 or w <= 0:
        return None

    actionable = {
        "door",
        "cabinet",
        "drawer",
        "refrigerator",
        "microwave",
        "oven",
        "toaster",
        "cup",
        "mug",
        "bottle",
        "glass",
        "wine glass",
        "can",
        "bowl",
        "plate",
        "box",
        "book",
        "phone",
        "remote",
        "chair",
    }
    drink_aliases = {"cup", "mug", "bottle", "glass", "wine glass", "water bottle"}

    best_name: str | None = None
    best_score = -1e9

    for b in action_boxes:
        raw_name = str(getattr(b, "class_name", "")).strip()
        if not raw_name:
            continue
        name = raw_name.lower().replace("_", " ").replace("-", " ").strip()
        token = name.replace(" ", "")
        conf = float(getattr(b, "confidence", 0.0) or 0.0)

        is_drink = (
            name in drink_aliases
            or "cup" in name
            or "mug" in name
            or "bottle" in name
            or token.endswith("glass")
            or name == "glass"
        )
        is_actionable = is_drink or (name in actionable)
        if not is_actionable:
            continue
        if conf < 0.35:
            continue

        x1 = float(getattr(b, "x1", 0.0))
        y1 = float(getattr(b, "y1", 0.0))
        x2 = float(getattr(b, "x2", 0.0))
        y2 = float(getattr(b, "y2", 0.0))
        bw = max(1.0, x2 - x1)
        bh = max(1.0, y2 - y1)
        area_ratio = max(0.0, min(1.0, (bw * bh) / float(w * h)))

        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        dx = (cx - (w * 0.5)) / max(1.0, w * 0.5)
        dy = (cy - (h * 0.5)) / max(1.0, h * 0.5)
        center_penalty = (dx * dx + dy * dy) ** 0.5

        class_priority = 3.0 if is_drink else 2.0 if name in {
            "door", "cabinet", "drawer", "refrigerator", "microwave", "oven", "toaster"
        } else 1.0

        score = (class_priority * 2.0) + conf + (area_ratio * 1.5) - (center_penalty * 0.8)
        if score > best_score:
            best_score = score
            best_name = raw_name

    if best_name is None or best_score < 2.2:
        return None
    return best_name


def _action_from_interaction(target_class: str, relation: str | None = None, contact_score: float | None = None) -> str:
    obj = (target_class or "object").strip().lower()
    rel = (relation or "").strip().lower()
    score = float(contact_score or 0.0)

    pickup_classes = {
        "cup",
        "mug",
        "bottle",
        "can",
        "bowl",
        "plate",
        "box",
        "book",
        "phone",
        "remote",
    }
    openable_classes = {
        "door",
        "cabinet",
        "drawer",
        "refrigerator",
        "microwave",
        "oven",
        "toaster",
    }

    if obj in pickup_classes:
        return f"TAKE {obj.upper()}" if score >= 0.2 else f"REACH {obj.upper()}"
    if obj in openable_classes:
        if rel in {"touching", "contact"} or score >= 0.2:
            return f"OPEN {obj.upper()}"
        return f"REACH {obj.upper()}"
    if obj in {"person", "scene", ""}:
        return "IDLE"
    if rel in {"touching", "contact"} or score >= 0.2:
        return f"INTERACT {obj.upper()}"
    return f"REACH {obj.upper()}"


class LiveVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.tracker = None
        self.segmenter = None
        self.detector = None
        self.vlm = None
        self.state_classifier = None
        self.interaction_engine = None
        self.seg_prompt = "person"
        self.vlm_interval = 12
        self.state_interval = 2
        self.state_max_boxes = 3
        self.state_min_conf = 0.25
        self.inference_max_side = 640
        self._frame_count = 0
        self._last_vlm_raw = ""
        self._last_vlm_parsed = None
        self._last_vlm_latency_s = None
        self._live_state_history: deque[str] = deque(maxlen=80)
        self._live_action_text: str | None = "IDLE"
        self._live_confidence: float | None = None
        self._live_action_hold: int = 0
        self._live_last_action_frame: int = 0
        self._live_no_action_cycles: int = 0
        self._state_target_classes = {
            "door",
            "cabinet",
            "drawer",
            "refrigerator",
            "microwave",
            "oven",
            "toaster",
            "cup",
            "mug",
            "bottle",
            "water bottle",
            "glass",
            "wine glass",
            "phone",
            "cell phone",
            "person",
        }

        self._record_lock = threading.Lock()
        self._recording = False
        self._record_path: str | None = None
        self._record_writer: cv2.VideoWriter | None = None
        self._record_fps = 20.0

    @property
    def recording(self) -> bool:
        with self._record_lock:
            return self._recording

    def start_recording(self, path: str, fps: float = 20.0) -> None:
        with self._record_lock:
            self._recording = True
            self._record_path = path
            self._record_fps = fps
            if self._record_writer is not None:
                self._record_writer.release()
                self._record_writer = None

    def stop_recording(self) -> str | None:
        with self._record_lock:
            self._recording = False
            if self._record_writer is not None:
                self._record_writer.release()
                self._record_writer = None
            path = self._record_path
        # Re-encode mp4v → H.264 for browser/OS playback
        if path and Path(path).exists() and Path(path).stat().st_size > 0:
            raw = Path(path)
            h264 = raw.with_suffix(".h264.mp4")
            _reencode_h264(raw, h264)
            raw.unlink(missing_ok=True)
            h264.rename(raw)
        return path

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self._frame_count += 1
        if self._live_action_hold > 0:
            self._live_action_hold -= 1

        with self._record_lock:
            if self._recording:
                if self._record_writer is None and self._record_path:
                    h, w = img.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    self._record_writer = cv2.VideoWriter(self._record_path, fourcc, self._record_fps, (w, h))
                if self._record_writer is not None:
                    self._record_writer.write(img)

        run_img = _resize_max_side(img, int(self.inference_max_side))
        classes_seen = []
        interaction_detection = None
        live_state_value = None
        live_state_conf = None
        state_eval_ran = False
        non_idle_action_detected = False

        if self.tracker is not None:
            trk = self.tracker.update(run_img)
            if trk.boxes:
                classes_seen = [b.class_name for b in trk.boxes]
                interaction_detection = DetectionResult(boxes=list(trk.boxes), frame_idx=self._frame_count)
                if self.state_classifier is None:
                    primary = _pick_primary_action_class(trk.boxes, run_img.shape[:2])
                    if primary and self._live_action_hold <= 0:
                        self._live_action_text = _action_from_object_state(primary, None)
                    elif self._live_action_hold <= 0:
                        self._live_action_text = "IDLE"
        elif self.detector is not None:
            det = self.detector.predict(run_img)
            if det.boxes:
                classes_seen = [b.class_name for b in det.boxes]
                interaction_detection = det
                if self.state_classifier is None:
                    primary = _pick_primary_action_class(det.boxes, run_img.shape[:2])
                    if primary and self._live_action_hold <= 0:
                        self._live_action_text = _action_from_object_state(primary, None)
                    elif self._live_action_hold <= 0:
                        self._live_action_text = "IDLE"

        if self.state_classifier is not None and (self._frame_count % max(1, int(self.state_interval)) == 0):
            state_eval_ran = True
            boxes = []
            box_names: list[str] = []
            if self.tracker is not None and interaction_detection and hasattr(interaction_detection, "boxes"):
                trk_boxes = [
                    b
                    for b in interaction_detection.boxes
                    if hasattr(b, "track_id") and _is_state_target_class(str(getattr(b, "class_name", "")))
                ]
                trk_boxes.sort(key=lambda b: float((b.x2 - b.x1) * (b.y2 - b.y1)), reverse=True)
                trk_boxes = trk_boxes[: max(1, int(self.state_max_boxes))]
                boxes = [
                    (
                        b.x1,
                        b.y1,
                        b.x2,
                        b.y2,
                        int(b.track_id),
                        str(getattr(b, "class_name", "object")),
                    )
                    for b in trk_boxes
                ]
                box_names = [str(getattr(b, "class_name", "object")) for b in trk_boxes]
            elif interaction_detection and hasattr(interaction_detection, "boxes"):
                filtered = [
                    b
                    for b in interaction_detection.boxes
                    if _is_state_target_class(str(getattr(b, "class_name", "")))
                ]
                filtered.sort(key=lambda b: float((b.x2 - b.x1) * (b.y2 - b.y1)), reverse=True)
                filtered = filtered[: max(1, int(self.state_max_boxes))]
                boxes = [
                    (
                        b.x1,
                        b.y1,
                        b.x2,
                        b.y2,
                        i,
                        str(getattr(b, "class_name", "object")),
                    )
                    for i, b in enumerate(filtered)
                ]
                box_names = [str(getattr(b, "class_name", "object")) for b in filtered]

            if boxes:
                try:
                    sc_out = self.state_classifier.classify(run_img, boxes)
                    def _state_rank(state_value: str) -> float:
                        sv = (state_value or "").strip().lower()
                        rank = {
                            "using_phone": 1.0,
                            "drinking_pepsi": 1.0,
                            "drinking": 1.0,
                            "wave": 0.9,
                            "point": 0.8,
                            "thumbs_up": 0.8,
                            "holding": 0.6,
                            "open": 0.5,
                            "ajar": 0.4,
                            "closed": 0.3,
                            "idle": 0.0,
                            "unknown": 0.0,
                        }
                        return rank.get(sv, 0.2)

                    def _object_rank(name_value: str) -> float:
                        nv = (name_value or "").strip().lower().replace("_", " ").replace("-", " ")
                        if "phone" in nv:
                            return 1.0
                        if any(k in nv for k in ("bottle", "cup", "mug", "glass", "wine glass")):
                            return 0.8
                        if nv == "person":
                            return 0.9
                        return 0.2

                    best_idx = 0
                    best_pair = sc_out[0] if sc_out else ("unknown", 0.0)
                    best_score = -1e9
                    best_non_idle_idx = -1
                    best_non_idle_conf = -1e9
                    best_non_idle_score = -1e9
                    for i, pair in enumerate(sc_out):
                        state_val = str(pair[0]) if len(pair) > 0 else "unknown"
                        conf_val = float(pair[1]) if len(pair) > 1 else 0.0
                        obj_name = box_names[i] if i < len(box_names) else "object"
                        score = conf_val + 0.35 * _state_rank(state_val) + 0.2 * _object_rank(obj_name)
                        if score > best_score:
                            best_score = score
                            best_idx = i
                            best_pair = pair
                        # Track strongest non-idle candidate independently, so idle boxes don't dominate.
                        action_probe = _action_from_object_state(obj_name, state_val)
                        if action_probe != "IDLE" and conf_val >= float(self.state_min_conf):
                            # Prefer a >= threshold non-idle candidate by confidence first,
                            # then score as tie-breaker.
                            if (conf_val > best_non_idle_conf) or (
                                abs(conf_val - best_non_idle_conf) < 1e-9 and score > best_non_idle_score
                            ):
                                best_non_idle_conf = conf_val
                                best_non_idle_score = score
                                best_non_idle_idx = i
                    if best_non_idle_idx >= 0:
                        best_idx = best_non_idle_idx
                        best_pair = sc_out[best_idx]
                    live_state_value = str(best_pair[0]).lower()
                    live_state_conf = float(best_pair[1])
                    if live_state_conf < float(self.state_min_conf):
                        live_state_value = None
                        live_state_conf = None
                    if box_names and live_state_value is not None:
                        obj_name = box_names[best_idx]
                        action_text = _action_from_object_state(obj_name, live_state_value)
                        if action_text != "IDLE":
                            # Always refresh non-idle actions immediately when they reappear.
                            self._live_action_text = action_text
                            self._live_confidence = live_state_conf
                            self._live_last_action_frame = self._frame_count
                            non_idle_action_detected = True
                            self._live_no_action_cycles = 0
                            if action_text in {"DRINK WATER", "DRINK PEPSI", "USE PHONE"}:
                                # Keep action stable briefly to avoid flicker.
                                self._live_action_hold = 4
                        elif self._live_action_hold <= 0:
                            self._live_action_text = "IDLE"
                            self._live_confidence = live_state_conf
                            self._live_last_action_frame = self._frame_count
                except Exception:
                    # Keep live banner usable even when a single SigLIP call fails.
                    if self._live_action_hold <= 0:
                        self._live_action_text = "IDLE"
                        self._live_confidence = None
                        self._live_last_action_frame = self._frame_count

        # If SigLIP ran this cycle and did not confirm a non-idle action, decay to IDLE after a short grace window.
        if self.state_classifier is not None and state_eval_ran and not non_idle_action_detected:
            self._live_no_action_cycles += 1
            if self._live_action_hold <= 0 and self._live_no_action_cycles >= 2:
                self._live_action_text = "IDLE"
                self._live_confidence = None
                self._live_last_action_frame = self._frame_count

        if self.segmenter is not None:
            seg = self.segmenter.predict(run_img, text_prompt=self.seg_prompt)
            if seg.masks:
                img = draw_segmentation(img, seg)

        if self.vlm is not None and (self._frame_count % max(1, int(self.vlm_interval)) == 0):
            prompt = _build_live_prompt(classes_seen)
            t0 = time.perf_counter()
            try:
                vlm_result = self.vlm.predict(run_img, prompt)
                self._last_vlm_latency_s = time.perf_counter() - t0
                raw_text = (vlm_result.raw_text or "").strip()
                parsed = vlm_result.parsed
                if not raw_text and not parsed:
                    # Retry once with a tiny deterministic prompt.
                    retry_prompt = (
                        'Return ONLY valid JSON: {"objects":[{"name":"str","state":"str","robot_actions":["str"]}]}'
                    )
                    vlm_result_retry = self.vlm.predict(run_img, retry_prompt)
                    raw_text = (vlm_result_retry.raw_text or "").strip()
                    parsed = vlm_result_retry.parsed
                if not raw_text and not parsed:
                    model_name = type(self.vlm).__name__
                    # Final fallback: synthesize minimal structured output from detector classes.
                    objects = [
                        {"name": c, "state": "unknown", "robot_actions": []}
                        for c in sorted(set(classes_seen))[:8]
                    ]
                    fallback = {"objects": objects, "error": "empty_vlm_output", "model": model_name}
                    self._last_vlm_raw = json.dumps(fallback, ensure_ascii=False)
                    self._last_vlm_parsed = fallback
                else:
                    self._last_vlm_raw = raw_text or json.dumps(parsed, ensure_ascii=False)
                    self._last_vlm_parsed = parsed

                # Build simple action/state signals for live HUD timeline.
                action_text = None
                confidence = None
                state_value = "unknown"
                if isinstance(self._last_vlm_parsed, dict):
                    objs = self._last_vlm_parsed.get("objects", [])
                    if isinstance(objs, list) and objs:
                        selected = None
                        for obj in objs:
                            if not isinstance(obj, dict):
                                continue
                            if str(obj.get("name", "object")).lower() != "person":
                                selected = obj
                                break
                        if selected is None and isinstance(objs[0], dict):
                            selected = objs[0]
                        if isinstance(selected, dict):
                            name = str(selected.get("name", "object"))
                            state_value = str(selected.get("state", "unknown")).lower()
                            actions = selected.get("robot_actions", [])
                            if str(name).lower() == "person":
                                name = "scene"
                            if isinstance(actions, list) and actions:
                                action_text = f"{str(actions[0]).upper()} {name}"
                            elif name and name != "scene":
                                action_text = _action_from_object_state(name, state_value)
                            conf_raw = selected.get("confidence")
                            if isinstance(conf_raw, (int, float)):
                                confidence = float(conf_raw)
                if self.state_classifier is None and self._live_action_hold <= 0:
                    self._live_action_text = action_text
                    self._live_confidence = confidence
                self._live_state_history.append(state_value)
            except Exception as exc:  # keep live overlay informative instead of silent blank
                self._last_vlm_latency_s = time.perf_counter() - t0
                model_name = type(self.vlm).__name__
                self._last_vlm_raw = f"VLM error ({model_name}): {exc}"
                self._last_vlm_parsed = {"error": str(exc), "model": model_name}

        if live_state_value is not None:
            self._live_state_history.append(live_state_value)
            if self._live_confidence is None and live_state_conf is not None:
                self._live_confidence = live_state_conf

        if self.vlm is not None:
            img = _overlay_vlm_summary(img, self._last_vlm_parsed, self._last_vlm_raw, self._last_vlm_latency_s)
        if self.interaction_engine is not None and self.interaction_engine.available:
            hand_poses, interactions = self.interaction_engine.process(run_img, interaction_detection)
            if hand_poses or interactions:
                temp_result = FrameResult(frame_idx=self._frame_count, frame=img, hand_poses=hand_poses, interactions=interactions)
                img = draw_hands_and_interactions(img, temp_result)
            # MediaPipe is used for hand/interactions visualization only.
            # Live action text is driven by SigLIP/VLM/object-state logic.

        if self.vlm is not None or self.state_classifier is not None:
            img = _overlay_live_task_timeline(
                img,
                action_text=self._live_action_text,
                confidence=self._live_confidence,
                state_history=self._live_state_history,
            )

        cv2.putText(img, f"Frame #{self._frame_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (220, 220, 220), 2)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


def _render_results(results: list[FrameResult], info: dict, nav_timeline: NavigationTimeline | None, key_prefix: str) -> None:
    timing_keys = sorted({k for r in results for k in r.timings.keys()})
    if timing_keys:
        st.caption("Inference timing (seconds)")
        timing_rows = []
        for key in timing_keys:
            vals = [r.timings[key] for r in results if key in r.timings]
            if vals:
                timing_rows.append(
                    {
                        "stage": key,
                        "mean_s": float(np.mean(vals)),
                        "p95_s": float(np.percentile(vals, 95)),
                        "max_s": float(np.max(vals)),
                    }
                )
        if timing_rows:
            timing_rows = sorted(timing_rows, key=lambda x: x["mean_s"], reverse=True)
            st.dataframe(timing_rows, width='stretch', hide_index=True)

    frame_idx = st.slider(
        "Frame",
        0,
        len(results) - 1,
        0,
        format=f"Frame %d / {len(results) - 1}",
        key=f"{key_prefix}_frame_slider",
    )
    r = results[frame_idx]
    hand_poses = getattr(r, "hand_poses", [])
    interactions = getattr(r, "interactions", [])

    tab_names = ["Navigation GT", "Original", "Combined"]
    if any(res.tracking for res in results):
        tab_names.append("Tracking")
    tab_names.append("Detections")
    tab_names.append("Segmentation")
    tab_names.append("Interactions")
    tab_names.append("VLM Output")
    if any(res.temporal_changes for res in results):
        tab_names.append("Temporal Changes")
    tab_names.extend(["Ground Truth JSON", "Accuracy"])

    tabs = st.tabs(tab_names)
    tab_map = {name: tab for name, tab in zip(tab_names, tabs)}

    with tab_map["Navigation GT"]:
        if nav_timeline and nav_timeline.objects:
            st.markdown("### Navigation State Timeline")
            mcols = st.columns(3)
            mcols[0].metric("Nav Objects Tracked", len(nav_timeline.objects))
            mcols[1].metric("State Transitions", nav_timeline.total_transitions)
            obj_types = set(o.object_type for o in nav_timeline.objects)
            mcols[2].metric("Object Types", ", ".join(sorted(obj_types)) if obj_types else "none")

            st.markdown("#### Per-Object State Timeline")
            state_colors = {
                "open": "#2ecc71",
                "closed": "#e74c3c",
                "ajar": "#f39c12",
                "blocked": "#9b59b6",
                "clear": "#3498db",
                "unknown": "#95a5a6",
            }
            total_frames = max(results[-1].frame_idx, 1)

            for obj in nav_timeline.objects:
                st.markdown(f"**{obj.name}** (Track #{obj.track_id}, type: {obj.object_type})")
                bar_html = '<div style="display:flex;height:28px;border-radius:4px;overflow:hidden;margin-bottom:8px;border:1px solid #555;">'
                for entry in obj.state_timeline:
                    width_pct = max((entry.frame_end - entry.frame_start) / total_frames * 100, 1.0)
                    color = state_colors.get(entry.state, "#95a5a6")
                    bar_html += (
                        f'<div style="width:{width_pct:.1f}%;background:{color};display:flex;align-items:center;justify-content:center;font-size:11px;color:#fff;font-weight:bold;min-width:20px;" '
                        f'title="frames {entry.frame_start}-{entry.frame_end}: {entry.state}">{entry.state}</div>'
                    )
                bar_html += "</div>"
                st.markdown(bar_html, unsafe_allow_html=True)

            st.markdown("#### State Transition Events")
            all_transitions = []
            for obj in nav_timeline.objects:
                for frame, from_s, to_s in obj.transitions:
                    all_transitions.append(
                        {
                            "Frame": frame,
                            "Time (s)": f"{frame / nav_timeline.video_fps:.2f}" if nav_timeline.video_fps > 0 else "-",
                            "Object": obj.name,
                            "Type": obj.object_type,
                            "Track ID": obj.track_id,
                            "From": from_s,
                            "To": to_s,
                        }
                    )
            if all_transitions:
                all_transitions.sort(key=lambda x: x["Frame"])
                st.dataframe(all_transitions, width='stretch')
            else:
                st.info("No state transitions detected.")
        else:
            st.info("No navigation objects detected.")

    with tab_map["Original"]:
        st.image(cv2.cvtColor(r.frame, cv2.COLOR_BGR2RGB), caption=f"Frame {r.frame_idx}", width='stretch')

    with tab_map["Combined"]:
        vis = draw_all(r.frame, r)
        st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"Combined - Frame {r.frame_idx}", width='stretch')

    if "Tracking" in tab_map:
        with tab_map["Tracking"]:
            if r.tracking:
                vis = draw_tracking(r.frame, r.tracking)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{len(r.tracking.boxes)} tracked objects", width='stretch')
            else:
                st.info("No tracking results for this frame.")

    with tab_map["Detections"]:
        if r.detection:
            vis = draw_detections(r.frame, r.detection)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{len(r.detection.boxes)} detections", width='stretch')
        else:
            st.info("No detection results for this frame.")

    with tab_map["Segmentation"]:
        if r.segmentation:
            vis = draw_segmentation(r.frame, r.segmentation)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption=f"{len(r.segmentation.masks)} masks", width='stretch')
        else:
            st.info("No segmentation results for this frame.")

    if "Interactions" in tab_map:
        with tab_map["Interactions"]:
            fps = float(info.get("fps", 0.0) or 0.0)
            any_hands = any(getattr(res, "hand_poses", []) for res in results)
            any_interactions = any(getattr(res, "interactions", []) for res in results)
            hand_frames = [res for res in results if getattr(res, "hand_poses", [])]
            interaction_frames = [res for res in results if getattr(res, "interactions", [])]
            st.caption(
                f"Frames with hands: {len(hand_frames)} / {len(results)} | "
                f"Frames with interactions: {len(interaction_frames)} / {len(results)}"
            )
            if hand_frames or interaction_frames:
                frame_rows = []
                for res in results:
                    res_hands = getattr(res, "hand_poses", [])
                    res_inter = getattr(res, "interactions", [])
                    if not res_hands and not res_inter:
                        continue
                    targets = sorted({f"{it.target_class}[{it.target_index}]" for it in res_inter})
                    frame_rows.append(
                        {
                            "frame_idx": res.frame_idx,
                            "time_s": round((res.frame_idx / fps), 3) if fps > 0 else None,
                            "hands": len(res_hands),
                            "interactions": len(res_inter),
                            "targets": ", ".join(targets),
                        }
                    )
                st.write("Frames with hands/interactions:")
                st.dataframe(frame_rows, width='stretch', hide_index=True)

            if hand_poses:
                st.dataframe(
                    [
                        {
                            "hand_id": h.hand_id,
                            "handedness": h.handedness,
                            "score": f"{h.score:.3f}",
                            "bbox": f"{h.bbox}",
                        }
                        for h in hand_poses
                    ]
                )
            else:
                st.info("No hands detected for this frame.")

            if interactions:
                st.dataframe(
                    [
                        {
                            "relation": inter.relation,
                            "hand": inter.hand_id,
                            "target": f"{inter.target_class}[{inter.target_index}]",
                            "contact_score": f"{inter.contact_score:.3f}",
                        }
                        for inter in interactions
                    ]
                )
            else:
                st.info("No interactions for this frame.")
            if not any_hands and not any_interactions:
                st.warning(
                    "No hand/object interaction data was produced for this run. "
                    "Enable interactions and verify MediaPipe is available."
                )

    with tab_map["VLM Output"]:
        if r.vlm:
            st.code(r.vlm.raw_text, language="text")
            if r.vlm.parsed:
                st.json(r.vlm.parsed)
        else:
            st.info("No VLM output for this frame.")
            vlm_frames = [
                res for res in results if res.vlm and (res.vlm.raw_text or res.vlm.parsed)
            ]
            if not vlm_frames:
                st.warning("No VLM output exists in this analyzed recording.")
            else:
                prev_candidates = [res for res in vlm_frames if res.frame_idx < r.frame_idx]
                next_candidates = [res for res in vlm_frames if res.frame_idx > r.frame_idx]
                prev_res = prev_candidates[-1] if prev_candidates else None
                next_res = next_candidates[0] if next_candidates else None
                if prev_res is not None:
                    with st.expander(f"Nearest previous VLM output (frame {prev_res.frame_idx})", expanded=True):
                        if prev_res.vlm is not None:
                            st.code(prev_res.vlm.raw_text, language="text")
                            if prev_res.vlm.parsed:
                                st.json(prev_res.vlm.parsed)
                if next_res is not None:
                    with st.expander(f"Nearest next VLM output (frame {next_res.frame_idx})", expanded=False):
                        if next_res.vlm is not None:
                            st.code(next_res.vlm.raw_text, language="text")
                            if next_res.vlm.parsed:
                                st.json(next_res.vlm.parsed)

    if "Temporal Changes" in tab_map:
        with tab_map["Temporal Changes"]:
            if r.temporal_changes:
                tc = r.temporal_changes
                st.markdown(f"**Comparing frames {tc.frame_idx_before} -> {tc.frame_idx_after}**")
                if tc.state_changes:
                    st.dataframe(
                        [
                            {
                                "Object": sc.object_name,
                                "Before": sc.before_state,
                                "After": sc.after_state,
                                "Confidence": sc.confidence,
                            }
                            for sc in tc.state_changes
                        ]
                    )
                vis = draw_temporal_changes(r.frame, tc)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB), caption="Temporal Changes Overlay", width='stretch')
            else:
                st.info("No temporal changes for this frame.")

    with tab_map["Ground Truth JSON"]:
        st.markdown("**Full Video Summary:**")
        gt = export_ground_truth(results, info, nav_timeline=nav_timeline)
        st.json(gt)
        gt_json = export_ground_truth_json(results, info, nav_timeline=nav_timeline)
        analysis_json = _build_analysis_results_json(results, info, nav_timeline=nav_timeline)
        dl_col1, dl_col2 = st.columns(2)
        with dl_col1:
            st.download_button(
                "Download Ground Truth JSON",
                gt_json,
                file_name="ground_truth.json",
                mime="application/json",
                type="primary",
                key=f"{key_prefix}_download_gt",
            )
        with dl_col2:
            st.download_button(
                "Download Analysis Results JSON",
                analysis_json,
                file_name="analysis_results.json",
                mime="application/json",
                key=f"{key_prefix}_download_results",
            )

    with tab_map["Accuracy"]:
        st.markdown("### Accuracy Measurement")
        if not nav_timeline or not nav_timeline.objects:
            st.warning("Run with Tracker + VLM first to generate predictions.")
        else:
            uploaded_labels = st.file_uploader(
                "Upload labels JSON",
                type=["json"],
                help='Format: [{"frame": 0, "object": "door", "state": "closed"}, ...]',
                key=f"{key_prefix}_labels_upload",
            )

            with st.expander("Or enter labels manually"):
                num_labels = st.number_input("Number of labels", 1, 50, 5, key=f"{key_prefix}_num_labels")
                manual_entries = []
                for i in range(int(num_labels)):
                    lcols = st.columns(3)
                    frame_num = lcols[0].number_input("Frame", 0, 100000, 0, key=f"{key_prefix}_lbl_frame_{i}")
                    obj_name = lcols[1].text_input("Object", value="door", key=f"{key_prefix}_lbl_obj_{i}")
                    obj_state = lcols[2].selectbox(
                        "State",
                        ["closed", "open", "ajar", "blocked", "clear", "unknown"],
                        key=f"{key_prefix}_lbl_state_{i}",
                    )
                    manual_entries.append({"frame": frame_num, "object": obj_name, "state": obj_state})

            eval_btn = st.button("Evaluate Accuracy", type="primary", key=f"{key_prefix}_eval_btn")
            if eval_btn:
                if uploaded_labels:
                    try:
                        labels_data = json.load(uploaded_labels)
                        manual_labels = parse_manual_labels(labels_data)
                    except json.JSONDecodeError:
                        st.error("Invalid JSON file.")
                        manual_labels = {}
                else:
                    manual_labels = parse_manual_labels(manual_entries)

                if not manual_labels:
                    st.error("No labels provided.")
                else:
                    metrics = evaluate_nav_accuracy(nav_timeline, manual_labels)
                    rcols = st.columns(3)
                    rcols[0].metric("Overall Accuracy", f"{metrics['accuracy']:.1%}")
                    rcols[1].metric("Correct", f"{metrics['correct']} / {metrics['total_labels']}")
                    rcols[2].metric("Labels Evaluated", metrics["total_labels"])
                    if metrics["details"]:
                        st.dataframe(metrics["details"], width='stretch')

    st.divider()
    export_btn = st.button("Export Annotated Video", key=f"{key_prefix}_export_btn")
    if export_btn:
        with st.spinner("Rendering video..."):
            out_path = Path(tempfile.mktemp(suffix=".mp4"))
            export_annotated_video(results, out_path, info["fps"], draw_all)
            st.session_state["live_exported_video"] = out_path.read_bytes()
            out_path.unlink(missing_ok=True)

    if st.session_state.get("live_exported_video") is not None:
        st.video(st.session_state["live_exported_video"], format="video/mp4")
        st.download_button(
            "Download MP4",
            st.session_state["live_exported_video"],
            file_name="annotated_output.mp4",
            mime="video/mp4",
            key=f"{key_prefix}_download_annotated",
        )


def _analyze_recording(
    record_path: str,
    *,
    use_interactions: bool,
    interaction_max_hands: int,
    interaction_contact_thresh: float,
    interaction_min_detect_conf: float,
    interaction_min_track_conf: float,
    frame_skip: int,
    vlm_skip: int,
    max_frames: int,
    sequential_offload: bool,
    enable_temporal_diff: bool,
    inference_max_side: int,
    vlm_prompt: str,
    enable_vlm: bool,
    seg_prompt: str,
    enable_seg: bool,
    enable_state_classifier: bool,
) -> None:
    path = Path(record_path)
    if not path.exists() or path.stat().st_size <= 0:
        st.warning("Selected recording is missing or empty.")
        return

    st.session_state["live_recording_path"] = str(path)
    st.session_state["live_recorded_video"] = path.read_bytes()
    st.info(f"Saved recording to: `{path.resolve()}`")

    with st.status("Post-processing recorded video...", expanded=True) as status:
        status.write("Reading recorded video metadata...")
        info = get_video_info(str(path))
        status.write(
            f"Loaded video: {info.get('frame_count', 0)} frames @ {info.get('fps', 0):.2f} FPS"
        )
        interaction_engine = None
        if use_interactions:
            status.write("Initializing MediaPipe interactions...")
            interaction_engine = InteractionEngine(
                InteractionConfig(
                    enabled=True,
                    max_hands=interaction_max_hands,
                    contact_threshold=interaction_contact_thresh,
                    min_detect_conf=interaction_min_detect_conf,
                    min_track_conf=interaction_min_track_conf,
                )
            )
            if not interaction_engine.available:
                detail = f" Details: {interaction_engine.error}" if interaction_engine.error else ""
                st.warning("MediaPipe interactions requested but unavailable." + detail)
                interaction_engine = None

        status.write("Building analysis pipeline...")
        pipeline = Pipeline(
            detector=st.session_state["live_detector"],
            segmenter=st.session_state["live_segmenter"],
            vlm=st.session_state["live_vlm"],
            tracker=st.session_state["live_tracker"],
            state_classifier=st.session_state["live_state_classifier"] if enable_state_classifier else None,
            vlm_prompt=(vlm_prompt if enable_vlm else ""),
            seg_prompt=(seg_prompt if enable_seg else ""),
            frame_skip=frame_skip,
            vlm_skip=vlm_skip,
            sequential_offload=sequential_offload,
            interaction_engine=interaction_engine,
            enable_temporal_diff=enable_temporal_diff,
            inference_max_side=(None if inference_max_side == 0 else int(inference_max_side)),
        )

        results: list[FrameResult] = []
        total = min(info["frame_count"], int(max_frames))
        status.write("Running inference...")
        progress = st.progress(0, text="Processing recorded frames...")
        denom = max(1, (total // max(1, frame_skip)) + 1)

        for result in pipeline.run(str(path), max_frames=int(max_frames)):
            results.append(result)
            pct = len(results) / denom
            progress.progress(min(pct, 1.0), text=f"Frame {result.frame_idx}/{total}")

        progress.progress(1.0, text="Done")
        st.session_state["live_results"] = results
        st.session_state["live_video_info"] = info
        st.session_state["live_nav_timeline"] = build_nav_timeline(results, fps=info["fps"]) if results else None
        st.session_state["live_last_analyzed_path"] = str(path)
        status.write(f"Post-processing complete. Generated {len(results)} analyzed frames.")
        status.update(label="Post-processing complete", state="complete")

    st.success("Analysis complete.")


_init_state()

with st.sidebar:
    st.header("Live Config")

    st.subheader("Tracking")
    enable_tracker = st.checkbox("Enable Tracker", value=True)
    tracker_names = list_models("tracker")
    trk_default = tracker_names.index(st.session_state["live_trk_name"]) if st.session_state["live_trk_name"] in tracker_names else 0
    trk_choice = st.selectbox("Tracker", tracker_names, index=trk_default, disabled=not enable_tracker)
    trk_kwargs = _pick_model_size("tracker", trk_choice, "Tracker Size") if enable_tracker else {}

    st.divider()
    enable_seg = st.checkbox("Enable Segmentation", value=True)
    segmenter_names = list_models("segmenter")
    seg_default = segmenter_names.index(st.session_state["live_seg_name"]) if st.session_state["live_seg_name"] in segmenter_names else 0
    seg_choice = st.selectbox("Segmenter", segmenter_names, index=seg_default, disabled=not enable_seg)
    seg_kwargs = _pick_model_size("segmenter", seg_choice, "Segmenter Size") if enable_seg else {}
    seg_prompt = st.text_input("Segmentation Prompt", value="person, door", disabled=not enable_seg)

    st.divider()
    enable_det = st.checkbox("Enable Detection", value=True)
    detector_names = list_models("detector")
    det_default = detector_names.index(st.session_state["live_det_name"]) if st.session_state["live_det_name"] in detector_names else 0
    det_choice = st.selectbox("Detector", detector_names, index=det_default, disabled=not enable_det)
    det_kwargs = _pick_model_size("detector", det_choice, "Detector Size") if enable_det else {}

    st.divider()
    enable_vlm = st.checkbox("Enable VLM", value=True)
    vlm_names = list_models("vlm")
    preferred = "gemini3" if "gemini3" in vlm_names else vlm_names[0]
    vlm_default = vlm_names.index(st.session_state["live_vlm_name"]) if st.session_state["live_vlm_name"] in vlm_names else vlm_names.index(preferred)
    vlm_choice = st.selectbox("VLM", vlm_names, index=vlm_default, disabled=not enable_vlm)
    prompt_mode = st.selectbox("Prompt Mode", ["Navigation (focused)", "Generic (broad)"], index=0, disabled=not enable_vlm)
    vlm_prompt = NAV_STATE_PROMPT if prompt_mode == "Navigation (focused)" else GENERIC_JSON_SCHEMA_PROMPT
    vlm_interval = st.slider("Run VLM every N frames", min_value=1, max_value=60, value=12, disabled=not enable_vlm)

    st.divider()
    enable_state_classifier = st.checkbox(
        "Enable State Classifier (SigLIP)",
        value=True,
        help="Fast per-object state labels (open/closed/ajar/etc).",
    )
    state_classifier_names = list_models("state_classifier")
    sc_default = (
        state_classifier_names.index(st.session_state["live_state_classifier_name"])
        if st.session_state["live_state_classifier_name"] in state_classifier_names
        else 0
    )
    state_classifier_choice = st.selectbox(
        "State Classifier",
        state_classifier_names,
        index=sc_default,
        disabled=not enable_state_classifier,
    )
    state_interval = st.slider(
        "Run State Classifier every N frames",
        min_value=1,
        max_value=30,
        value=2,
        disabled=not enable_state_classifier,
    )
    state_max_boxes = st.slider(
        "Max boxes per state pass",
        min_value=1,
        max_value=8,
        value=3,
        disabled=not enable_state_classifier,
    )
    state_min_conf = st.slider(
        "Min state confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.25,
        step=0.05,
        disabled=not enable_state_classifier,
    )

    st.divider()
    inference_max_side = st.select_slider(
        "Inference Max Side (px)",
        options=[0, 320, 480, 640, 720, 960, 1280],
        value=960,
        help="0 disables resizing.",
    )

    st.subheader("Post-Stop Analysis")
    frame_skip = st.slider("Process every N frames", 1, 30, 1)
    vlm_skip = st.slider("VLM every N processed frames", 1, 50, 5)
    max_frames = st.number_input("Max frames to process", 10, 100000, 10000)
    sequential_offload = st.checkbox("Sequential Offload (low VRAM)", value=False)
    enable_temporal_diff = st.checkbox("Enable Temporal Diff", value=True)
    st.subheader("Interactions")
    use_interactions = st.checkbox("Enable hand-object interactions (MediaPipe)", value=True)
    interaction_max_hands = st.slider("Max hands", 1, 4, 2, disabled=not use_interactions)
    interaction_contact_thresh = st.slider(
        "Contact threshold",
        min_value=0.05,
        max_value=0.9,
        value=0.1,
        step=0.05,
        disabled=not use_interactions,
    )
    interaction_min_detect_conf = st.slider(
        "Hand detect confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        disabled=not use_interactions,
    )
    interaction_min_track_conf = st.slider(
        "Hand track confidence",
        min_value=0.05,
        max_value=0.95,
        value=0.35,
        step=0.05,
        disabled=not use_interactions,
    )

    load_btn = st.button("Load Selected", type="primary", width='stretch')
    unload_btn = st.button("Unload All", width='stretch')

    if not st.session_state["live_loaded"] and not st.session_state["live_auto_loaded"]:
        with st.spinner("Auto-loading defaults..."):
            _load_selected_models(
                enable_tracker=enable_tracker,
                enable_seg=enable_seg,
                enable_det=enable_det,
                enable_vlm=enable_vlm,
                enable_state_classifier=enable_state_classifier,
                trk_choice=trk_choice,
                seg_choice=seg_choice,
                det_choice=det_choice,
                vlm_choice=vlm_choice,
                state_classifier_choice=state_classifier_choice,
                trk_kwargs=trk_kwargs,
                seg_kwargs=seg_kwargs,
                det_kwargs=det_kwargs,
            )
        st.session_state["live_auto_loaded"] = True
        st.success("Default models auto-loaded.")

    if load_btn:
        with st.spinner("Loading selected models..."):
            _load_selected_models(
                enable_tracker=enable_tracker,
                enable_seg=enable_seg,
                enable_det=enable_det,
                enable_vlm=enable_vlm,
                enable_state_classifier=enable_state_classifier,
                trk_choice=trk_choice,
                seg_choice=seg_choice,
                det_choice=det_choice,
                vlm_choice=vlm_choice,
                state_classifier_choice=state_classifier_choice,
                trk_kwargs=trk_kwargs,
                seg_kwargs=seg_kwargs,
                det_kwargs=det_kwargs,
            )
        st.success("Models loaded.")

    if unload_btn:
        _safe_unload(st.session_state["live_tracker"])
        _safe_unload(st.session_state["live_segmenter"])
        _safe_unload(st.session_state["live_detector"])
        _safe_unload(st.session_state["live_vlm"])
        _safe_unload(st.session_state["live_state_classifier"])
        st.session_state["live_tracker"] = None
        st.session_state["live_segmenter"] = None
        st.session_state["live_detector"] = None
        st.session_state["live_vlm"] = None
        st.session_state["live_state_classifier"] = None
        st.session_state["live_loaded"] = False
        st.info("All models unloaded.")

    loaded = []
    if st.session_state["live_tracker"] is not None:
        loaded.append(f"trk:{st.session_state['live_trk_name']}")
    if st.session_state["live_segmenter"] is not None:
        loaded.append(f"seg:{st.session_state['live_seg_name']}")
    if st.session_state["live_detector"] is not None:
        loaded.append(f"det:{st.session_state['live_det_name']}")
    if st.session_state["live_vlm"] is not None:
        loaded.append(f"vlm:{st.session_state['live_vlm_name']}")
    if st.session_state["live_state_classifier"] is not None:
        loaded.append(f"state:{st.session_state['live_state_classifier_name']}")
    st.caption("Loaded: " + (", ".join(loaded) if loaded else "none"))

    if use_interactions:
        if (
            st.session_state["live_interaction_engine"] is None
            or not st.session_state["live_interaction_engine"].config.enabled
            or st.session_state["live_interaction_engine"].config.max_hands != interaction_max_hands
            or abs(st.session_state["live_interaction_engine"].config.contact_threshold - interaction_contact_thresh) > 1e-9
            or abs(st.session_state["live_interaction_engine"].config.min_detect_conf - interaction_min_detect_conf) > 1e-9
            or abs(st.session_state["live_interaction_engine"].config.min_track_conf - interaction_min_track_conf) > 1e-9
        ):
            old_engine = st.session_state["live_interaction_engine"]
            if old_engine is not None:
                old_engine.close()
            st.session_state["live_interaction_engine"] = InteractionEngine(
                InteractionConfig(
                    enabled=True,
                    max_hands=interaction_max_hands,
                    contact_threshold=interaction_contact_thresh,
                    min_detect_conf=interaction_min_detect_conf,
                    min_track_conf=interaction_min_track_conf,
                )
            )
        live_ie = st.session_state["live_interaction_engine"]
        if live_ie is not None and not live_ie.available:
            detail = f" Details: {live_ie.error}" if live_ie.error else ""
            st.warning("MediaPipe interactions requested but unavailable." + detail)
    else:
        old_engine = st.session_state["live_interaction_engine"]
        if old_engine is not None:
            old_engine.close()
        st.session_state["live_interaction_engine"] = None


if not st.session_state["live_loaded"]:
    st.warning("Load at least one model from the sidebar before starting the stream.")
else:
    ctx = webrtc_streamer(
        key="live-perception",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=LiveVideoProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # Recover from stale state when user navigates away and returns.
    if not ctx.video_processor:
        if st.session_state.get("live_prev_playing", False):
            st.session_state["live_prev_playing"] = False
        if st.session_state.get("live_recording", False):
            st.session_state["live_recording"] = False
            st.info("Recovered live session state after page navigation. Start the stream again.")

    if ctx.video_processor:
        ctx.video_processor.tracker = st.session_state["live_tracker"]
        ctx.video_processor.segmenter = st.session_state["live_segmenter"]
        ctx.video_processor.detector = st.session_state["live_detector"]
        ctx.video_processor.vlm = st.session_state["live_vlm"] if enable_vlm else None
        ctx.video_processor.state_classifier = (
            st.session_state["live_state_classifier"] if enable_state_classifier else None
        )
        ctx.video_processor.interaction_engine = st.session_state["live_interaction_engine"]
        ctx.video_processor.seg_prompt = seg_prompt
        ctx.video_processor.vlm_interval = vlm_interval
        ctx.video_processor.state_interval = state_interval
        ctx.video_processor.state_max_boxes = state_max_boxes
        ctx.video_processor.state_min_conf = state_min_conf
        ctx.video_processor.inference_max_side = inference_max_side

        is_playing = bool(getattr(getattr(ctx, "state", None), "playing", False))

        c1, c2, c3 = st.columns(3)
        c1.write("Recording auto-starts when live stream is active.")
        c2.write("Use WebRTC START/STOP above.")
        st.caption("Press WebRTC `STOP` to stop stream, then automatic post-processing runs on the saved clip.")

        if is_playing and not ctx.video_processor.recording and not st.session_state["live_recording"]:
            captures_dir = Path("outputs/live_captures")
            captures_dir.mkdir(parents=True, exist_ok=True)
            ts = time.strftime("%Y%m%d_%H%M%S")
            record_path = str(captures_dir / f"live_capture_{ts}.mp4")
            ctx.video_processor.start_recording(record_path, fps=20.0)
            st.session_state["live_recording"] = True
            st.session_state["live_recording_path"] = record_path
            st.info(f"Recording started automatically: `{Path(record_path).resolve()}`")

        just_stopped = st.session_state.get("live_prev_playing", False) and not is_playing
        st.session_state["live_prev_playing"] = is_playing

        if just_stopped and st.session_state["live_recording"]:
            record_path = ctx.video_processor.stop_recording()
            st.session_state["live_recording"] = False
            st.session_state["live_recording_path"] = record_path
            if (
                record_path
                and Path(record_path).exists()
                and Path(record_path).stat().st_size > 0
                and st.session_state.get("live_last_analyzed_path") != record_path
            ):
                _analyze_recording(
                    record_path,
                    use_interactions=use_interactions,
                    interaction_max_hands=interaction_max_hands,
                    interaction_contact_thresh=interaction_contact_thresh,
                    interaction_min_detect_conf=interaction_min_detect_conf,
                    interaction_min_track_conf=interaction_min_track_conf,
                    frame_skip=frame_skip,
                    vlm_skip=vlm_skip,
                    max_frames=int(max_frames),
                    sequential_offload=sequential_offload,
                    enable_temporal_diff=enable_temporal_diff,
                    inference_max_side=inference_max_side,
                    vlm_prompt=vlm_prompt,
                    enable_vlm=enable_vlm,
                    seg_prompt=seg_prompt,
                    enable_seg=enable_seg,
                    enable_state_classifier=enable_state_classifier,
                )
            else:
                st.warning("No recorded video found. Start recording first, then stop.")

        c3.write("Recording: **ON**" if ctx.video_processor.recording else "Recording: **OFF**")


if st.session_state.get("live_recorded_video"):
    st.subheader("Recorded Video")
    if st.session_state.get("live_recording_path"):
        st.caption(f"Saved path: `{Path(st.session_state['live_recording_path']).resolve()}`")
    st.video(st.session_state["live_recorded_video"], format="video/mp4")
    st.download_button(
        "Download Recorded MP4",
        st.session_state["live_recorded_video"],
        file_name="live_capture.mp4",
        mime="video/mp4",
        key="download_recorded_live",
    )

captures_dir = Path("outputs/live_captures")
history_files = sorted(captures_dir.glob("*.mp4"), key=lambda p: p.stat().st_mtime, reverse=True) if captures_dir.exists() else []
if history_files:
    st.subheader("Recording History")
    selected_name = st.selectbox(
        "Open previous recording",
        options=[p.name for p in history_files],
        key="live_history_select",
    )
    selected_path = captures_dir / selected_name
    if selected_path.exists():
        selected_bytes = selected_path.read_bytes()
        st.caption(f"History path: `{selected_path.resolve()}`")
        st.video(selected_bytes, format="video/mp4")
        st.download_button(
            "Download Selected Recording",
            selected_bytes,
            file_name=selected_path.name,
            mime="video/mp4",
            key="download_selected_live_history",
        )
        if st.button("Analyze Selected Recording", key="analyze_selected_live_history", type="primary"):
            _analyze_recording(
                str(selected_path),
                use_interactions=use_interactions,
                interaction_max_hands=interaction_max_hands,
                interaction_contact_thresh=interaction_contact_thresh,
                interaction_min_detect_conf=interaction_min_detect_conf,
                interaction_min_track_conf=interaction_min_track_conf,
                frame_skip=frame_skip,
                vlm_skip=vlm_skip,
                max_frames=int(max_frames),
                sequential_offload=sequential_offload,
                enable_temporal_diff=enable_temporal_diff,
                inference_max_side=inference_max_side,
                vlm_prompt=vlm_prompt,
                enable_vlm=enable_vlm,
                seg_prompt=seg_prompt,
                enable_seg=enable_seg,
                enable_state_classifier=enable_state_classifier,
            )

live_results = st.session_state.get("live_results", [])
if live_results:
    st.divider()
    st.subheader("Post-Stop Analysis")
    _render_results(
        live_results,
        st.session_state["live_video_info"],
        st.session_state.get("live_nav_timeline"),
        key_prefix="live",
    )
