from __future__ import annotations

import tempfile
import json
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import models  # triggers registration
from core.registry import create, get_class, list_models
from core.video import get_video_info, export_annotated_video
from core.types import FrameResult, NavigationTimeline
from models.vlm.lfm25 import NAV_STATE_PROMPT, GENERIC_JSON_SCHEMA_PROMPT
from pipeline.runner import Pipeline
from pipeline.interactions import InteractionConfig, InteractionEngine
from pipeline.visualizer import (
    draw_detections, draw_segmentation, draw_vlm_text, draw_all,
    draw_tracking, draw_temporal_changes,
)
from pipeline.exporter import export_ground_truth, export_ground_truth_json
from pipeline.nav_state import build_nav_timeline
from pipeline.evaluator import evaluate_nav_accuracy, parse_manual_labels

st.set_page_config(layout="wide", page_title="World2Data — Navigation Ground Truth")


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

# --- Session state defaults ---
for key, default in [
    ("results", []),
    ("models_loaded", False),
    ("detector_instance", None),
    ("segmenter_instance", None),
    ("vlm_instance", None),
    ("tracker_instance", None),
    ("state_classifier_instance", None),
    ("exported_video", None),
    ("nav_timeline", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# --- Display size CSS ---
fullscreen = st.checkbox("Fullscreen images", value=False)
if not fullscreen:
    st.markdown(
        """<style>
        img, video { max-height: 500px !important; object-fit: contain !important; }
        </style>""",
        unsafe_allow_html=True,
    )

# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.header("Model Configuration")

    # --- Tracker (default ON for nav) ---
    st.subheader("Object Tracking")
    use_tracker = st.checkbox("Enable Tracker", value=True,
                              help="Multi-object tracking with persistent IDs across frames")
    if use_tracker:
        tracker_names = list_models("tracker")
        tracker_choice = st.selectbox("Tracker Model", tracker_names, index=0)
        tracker_size = st.selectbox("Tracker Model Size", ["n", "s", "m", "l", "x"], index=1,
                                    key="tracker_size")

    # --- State Classifier (default ON for nav) ---
    st.subheader("State Classifier")
    use_state_classifier = st.checkbox("Enable State Classifier", value=True,
                                       help="SigLIP zero-shot state classification on tracked objects (~50ms/crop vs ~10s VLM)")
    if use_state_classifier:
        sc_names = list_models("state_classifier")
        sc_choice = st.selectbox("State Classifier Model", sc_names, index=0)

    # --- Detector ---
    st.subheader("Object Detection")
    use_detector = st.checkbox("Enable Detector", value=not use_tracker,
                               help="Standalone detection (redundant if tracker is enabled)")
    if use_detector:
        det_names = list_models("detector")
        det_choice = st.selectbox("Detector Model", det_names, index=0)
        det_cls = get_class("detector", det_choice)
        det_sizes = list(det_cls.VALID_SIZES)
        det_size = st.selectbox("Model Size", det_sizes, index=0)

    # --- Segmenter ---
    st.subheader("Segmentation")
    use_segmenter = st.checkbox("Enable Segmenter", value=False)
    if use_segmenter:
        seg_names = list_models("segmenter")
        seg_choice = st.selectbox("Segmenter Model", seg_names, index=0)
        seg_cls = get_class("segmenter", seg_choice)
        seg_size = None
        if hasattr(seg_cls, "VALID_SIZES"):
            seg_sizes = list(seg_cls.VALID_SIZES)
            seg_size = st.selectbox("Segmenter Size", seg_sizes, index=0)
        seg_prompt = st.text_input("Segmentation Text Prompt", value="door, drawer, handle",
                                   help="Open-vocabulary: what to segment")

    # --- VLM (default ON for nav) ---
    st.subheader("Vision-Language Model")
    use_vlm = st.checkbox("Enable VLM", value=True)
    if use_vlm:
        vlm_names = list_models("vlm")
        preferred_vlm = "gemini3" if "gemini3" in vlm_names else vlm_names[0]
        vlm_choice = st.selectbox("VLM Model", vlm_names, index=vlm_names.index(preferred_vlm))
        prompt_mode = st.selectbox("Prompt Mode", ["Navigation (focused)", "Generic (broad)"], index=0)
        if prompt_mode == "Navigation (focused)":
            vlm_prompt = st.text_area("VLM Prompt", value=NAV_STATE_PROMPT, height=200)
        else:
            vlm_prompt = st.text_area("VLM Prompt", value=GENERIC_JSON_SCHEMA_PROMPT, height=200)
        enable_temporal_diff = st.checkbox(
            "Enable Temporal Diff",
            value=True,
            help="Compare consecutive VLM frames to detect state changes over time",
        )
    else:
        enable_temporal_diff = False

    # --- Optional interactions ---
    max_hands = 2
    contact_threshold = 0.2
    st.subheader("Interactions (Optional)")
    use_interactions = st.checkbox(
        "Enable hand-object interactions",
        value=False,
        help="Uses MediaPipe Hands and links hands to any detected object boxes.",
    )
    if use_interactions:
        max_hands = st.slider("Max hands", 1, 4, 2)
        contact_threshold = st.slider("Contact threshold", 0.05, 0.95, 0.2, 0.05)

    # --- Pipeline settings ---
    st.subheader("Pipeline Settings")
    frame_skip = st.slider("Process every N frames", 1, 30, 1)
    vlm_skip = st.slider("VLM: run every N processed frames", 1, 50, 5,
                          help="VLM is slow. This skips frames independently of the main frame skip.")
    inference_max_side = st.select_slider(
        "Inference Max Side (px)",
        options=[0, 320, 480, 640, 720, 960, 1280],
        value=640,
        help="Resize frames before inference to speed up processing. 0 = original resolution.",
    )
    max_frames = st.number_input("Max frames to process", 10, 100000, 750)
    sequential_offload = st.checkbox("Sequential Offload (low VRAM)",
                                     value=False,
                                     help="Load/unload each model per-frame. Slower but uses less VRAM.")

    st.divider()

    # --- Load / Unload ---
    col1, col2 = st.columns(2)
    with col1:
        load_btn = st.button("Load Models", type="primary", width='stretch')
    with col2:
        unload_btn = st.button("Unload All", width='stretch')

    if load_btn:
        with st.spinner("Loading models..."):
            # Tracker
            if use_tracker:
                trk = create("tracker", tracker_choice, model_size=tracker_size)
                if not sequential_offload:
                    trk.load()
                st.session_state["tracker_instance"] = trk
            else:
                st.session_state["tracker_instance"] = None

            # State Classifier
            if use_state_classifier:
                sc = create("state_classifier", sc_choice)
                if not sequential_offload:
                    sc.load()
                st.session_state["state_classifier_instance"] = sc
            else:
                st.session_state["state_classifier_instance"] = None

            # Detector
            if use_detector:
                det = create("detector", det_choice, model_size=det_size)
                if not sequential_offload:
                    det.load()
                st.session_state["detector_instance"] = det
            else:
                st.session_state["detector_instance"] = None

            # Segmenter
            if use_segmenter:
                seg_kwargs = {"model_size": seg_size} if seg_size else {}
                seg = create("segmenter", seg_choice, **seg_kwargs)
                if not sequential_offload:
                    seg.load()
                st.session_state["segmenter_instance"] = seg
            else:
                st.session_state["segmenter_instance"] = None

            # VLM
            if use_vlm:
                vlm = create("vlm", vlm_choice)
                if not sequential_offload:
                    vlm.load()
                st.session_state["vlm_instance"] = vlm
            else:
                st.session_state["vlm_instance"] = None

            st.session_state["models_loaded"] = True
        st.success("Models loaded!")

    if unload_btn:
        for key in ("detector_instance", "segmenter_instance", "vlm_instance",
                     "tracker_instance", "state_classifier_instance"):
            inst = st.session_state.get(key)
            if inst is not None:
                inst.unload()
            st.session_state[key] = None
        st.session_state["models_loaded"] = False
        st.session_state["results"] = []
        st.session_state["nav_timeline"] = None
        st.success("All models unloaded.")

    # Status
    st.divider()
    st.caption("Loaded models:")
    for label, key in [
        ("Tracker", "tracker_instance"),
        ("State Classifier", "state_classifier_instance"),
        ("Detector", "detector_instance"),
        ("Segmenter", "segmenter_instance"),
        ("VLM", "vlm_instance"),
    ]:
        inst = st.session_state.get(key)
        st.write(f"- {label}: {'✓ ' + type(inst).__name__ if inst else '—'}")


# ============================================================
# MAIN AREA
# ============================================================
st.title("World2Data — Navigation Ground Truth")
st.caption("AI-powered ground truth generation for humanoid navigation: doors, drawers, handles, passages, obstacles & their state transitions")

uploaded = st.file_uploader("Upload video", type=["mp4", "avi", "mov", "mkv"])

if uploaded:
    # Save to temp file
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    tmp.write(uploaded.read())
    tmp.flush()
    video_path = tmp.name

    # Show video info
    info = get_video_info(video_path)
    cols = st.columns(4)
    cols[0].metric("FPS", f"{info['fps']:.1f}")
    cols[1].metric("Frames", info["frame_count"])
    cols[2].metric("Width", info["width"])
    cols[3].metric("Height", info["height"])

    # Run inference
    run_btn = st.button("Run Inference", type="primary",
                        disabled=not st.session_state["models_loaded"])

    if run_btn:
        interaction_engine = None
        if use_interactions:
            interaction_engine = InteractionEngine(
                InteractionConfig(
                    enabled=True,
                    max_hands=max_hands,
                    contact_threshold=contact_threshold,
                )
            )
            if not interaction_engine.available:
                detail = f" ({interaction_engine.error})" if interaction_engine.error else ""
                st.warning(
                    "MediaPipe interactions requested but unavailable." + detail
                )

        pipeline = Pipeline(
            detector=st.session_state["detector_instance"],
            segmenter=st.session_state["segmenter_instance"],
            vlm=st.session_state["vlm_instance"],
            tracker=st.session_state["tracker_instance"],
            state_classifier=st.session_state["state_classifier_instance"],
            vlm_prompt=vlm_prompt if use_vlm else "",
            seg_prompt=seg_prompt if use_segmenter else "",
            frame_skip=frame_skip,
            vlm_skip=vlm_skip,
            sequential_offload=sequential_offload,
            interaction_engine=interaction_engine,
            enable_temporal_diff=enable_temporal_diff,
            inference_max_side=(None if inference_max_side == 0 else int(inference_max_side)),
        )

        results: list[FrameResult] = []
        total = min(info["frame_count"], max_frames)
        progress = st.progress(0, text="Processing frames...")
        frame_display = st.empty()

        for result in pipeline.run(video_path, max_frames=max_frames):
            results.append(result)
            pct = len(results) / (total // frame_skip + 1)
            progress.progress(min(pct, 1.0), text=f"Frame {result.frame_idx}/{total}")

            # Live preview every 5 processed frames
            if len(results) % 5 == 1:
                preview = draw_all(result.frame, result)
                frame_display.image(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB),
                                    caption=f"Frame {result.frame_idx}", width='stretch')

        progress.progress(1.0, text="Done!")
        st.session_state["results"] = results

        # Build navigation timeline
        nav_tl = build_nav_timeline(results, fps=info["fps"])
        st.session_state["nav_timeline"] = nav_tl

        st.success(f"Processed {len(results)} frames. Found {len(nav_tl.objects)} nav objects, {nav_tl.total_transitions} state transitions.")

    # --- Results viewer ---
    results = st.session_state["results"]
    nav_timeline: NavigationTimeline | None = st.session_state.get("nav_timeline")

    if results:
        st.divider()
        st.subheader("Results Viewer")

        # Timing summary across processed frames
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

        frame_idx = st.slider("Frame", 0, len(results) - 1, 0,
                               format=f"Frame %d / {len(results) - 1}")
        r = results[frame_idx]
        hand_poses = getattr(r, "hand_poses", [])
        interactions = getattr(r, "interactions", [])

        # Build tab list — Navigation GT is prominent
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

        # ============================================================
        # NAVIGATION GT TAB
        # ============================================================
        with tab_map["Navigation GT"]:
            if nav_timeline and nav_timeline.objects:
                st.markdown("### Navigation State Timeline")

                # Summary metrics
                mcols = st.columns(3)
                mcols[0].metric("Nav Objects Tracked", len(nav_timeline.objects))
                mcols[1].metric("State Transitions", nav_timeline.total_transitions)
                obj_types = set(o.object_type for o in nav_timeline.objects)
                mcols[2].metric("Object Types", ", ".join(sorted(obj_types)) if obj_types else "none")

                # State timeline visualization as colored bars
                st.markdown("#### Per-Object State Timeline")

                _STATE_COLORS = {
                    "open": "#2ecc71",
                    "closed": "#e74c3c",
                    "ajar": "#f39c12",
                    "blocked": "#9b59b6",
                    "clear": "#3498db",
                    "unknown": "#95a5a6",
                }

                total_frames = results[-1].frame_idx if results else 1

                for obj in nav_timeline.objects:
                    label = f"**{obj.name}** (Track #{obj.track_id}, type: {obj.object_type})"
                    st.markdown(label)

                    # Build HTML bar
                    bar_html = '<div style="display:flex;height:28px;border-radius:4px;overflow:hidden;margin-bottom:8px;border:1px solid #555;">'
                    for entry in obj.state_timeline:
                        width_pct = max(
                            (entry.frame_end - entry.frame_start) / total_frames * 100,
                            1.0
                        )
                        color = _STATE_COLORS.get(entry.state, "#95a5a6")
                        bar_html += (
                            f'<div style="width:{width_pct:.1f}%;background:{color};'
                            f'display:flex;align-items:center;justify-content:center;'
                            f'font-size:11px;color:#fff;font-weight:bold;min-width:20px;" '
                            f'title="frames {entry.frame_start}-{entry.frame_end}: {entry.state}">'
                            f'{entry.state}</div>'
                        )
                    bar_html += '</div>'
                    st.markdown(bar_html, unsafe_allow_html=True)

                # Legend
                legend_html = '<div style="display:flex;gap:12px;margin:8px 0 16px 0;flex-wrap:wrap;">'
                for state, color in _STATE_COLORS.items():
                    legend_html += f'<span style="display:flex;align-items:center;gap:4px;"><span style="width:14px;height:14px;background:{color};border-radius:3px;display:inline-block;"></span><span style="font-size:12px;">{state}</span></span>'
                legend_html += '</div>'
                st.markdown(legend_html, unsafe_allow_html=True)

                # Transition events table
                st.markdown("#### State Transition Events")
                all_transitions = []
                for obj in nav_timeline.objects:
                    for frame, from_s, to_s in obj.transitions:
                        all_transitions.append({
                            "Frame": frame,
                            "Time (s)": f"{frame / nav_timeline.video_fps:.2f}" if nav_timeline.video_fps > 0 else "—",
                            "Object": obj.name,
                            "Type": obj.object_type,
                            "Track ID": obj.track_id,
                            "From": from_s,
                            "To": to_s,
                        })
                if all_transitions:
                    all_transitions.sort(key=lambda x: x["Frame"])
                    st.dataframe(all_transitions, width='stretch')
                else:
                    st.info("No state transitions detected. Try processing more frames or adjusting VLM skip.")

                # Current frame state
                st.markdown(f"#### Current Frame State (Frame {r.frame_idx})")
                frame_states = []
                for obj in nav_timeline.objects:
                    for entry in obj.state_timeline:
                        if entry.frame_start <= r.frame_idx <= entry.frame_end:
                            frame_states.append({
                                "Object": obj.name,
                                "Type": obj.object_type,
                                "Track ID": obj.track_id,
                                "State": entry.state,
                            })
                if frame_states:
                    st.dataframe(frame_states, width='stretch')
                else:
                    st.info("No navigation objects have state data at this frame.")

            else:
                st.info("No navigation objects detected. Enable VLM with the Navigation prompt and Tracker, then re-run inference.")

        # ============================================================
        # ORIGINAL TAB
        # ============================================================
        with tab_map["Original"]:
            st.image(cv2.cvtColor(r.frame, cv2.COLOR_BGR2RGB),
                     caption=f"Frame {r.frame_idx}", width='stretch')

        # ============================================================
        # COMBINED TAB
        # ============================================================
        with tab_map["Combined"]:
            vis = draw_all(r.frame, r)
            st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                     caption=f"Combined — Frame {r.frame_idx}", width='stretch')
            if r.timings:
                st.caption("Per-frame timing (seconds)")
                st.json({k: round(v, 4) for k, v in r.timings.items()})

        # ============================================================
        # TRACKING TAB
        # ============================================================
        if "Tracking" in tab_map:
            with tab_map["Tracking"]:
                if r.tracking:
                    vis = draw_tracking(r.frame, r.tracking)
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                             caption=f"{len(r.tracking.boxes)} tracked objects",
                             width='stretch')
                    if r.tracking.boxes:
                        st.dataframe([
                            {"track_id": b.track_id, "class": b.class_name,
                             "confidence": f"{b.confidence:.3f}",
                             "bbox": f"({int(b.x1)},{int(b.y1)},{int(b.x2)},{int(b.y2)})"}
                            for b in r.tracking.boxes
                        ])
                else:
                    st.info("No tracking results for this frame.")

        # ============================================================
        # DETECTIONS TAB
        # ============================================================
        with tab_map["Detections"]:
            if r.detection:
                vis = draw_detections(r.frame, r.detection)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                         caption=f"{len(r.detection.boxes)} detections", width='stretch')
                if r.detection.boxes:
                    st.dataframe([
                        {"class": b.class_name, "confidence": f"{b.confidence:.3f}",
                         "bbox": f"({int(b.x1)},{int(b.y1)},{int(b.x2)},{int(b.y2)})"}
                        for b in r.detection.boxes
                    ])
            else:
                st.info("No detection results for this frame.")

        # ============================================================
        # SEGMENTATION TAB
        # ============================================================
        with tab_map["Segmentation"]:
            if r.segmentation:
                vis = draw_segmentation(r.frame, r.segmentation)
                st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                         caption=f"{len(r.segmentation.masks)} masks", width='stretch')
            else:
                st.info("No segmentation results for this frame.")

        # ============================================================
        # INTERACTIONS TAB
        # ============================================================
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
                    st.write(f"Hands: {len(hand_poses)}")
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
                    st.write("Interactions:")
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
                    st.info("No interactions found for this frame.")
                if not any_hands and not any_interactions:
                    st.warning(
                        "No hand/object interaction data was produced for this run. "
                        "Enable interactions and verify MediaPipe is available."
                    )

        # ============================================================
        # VLM OUTPUT TAB
        # ============================================================
        with tab_map["VLM Output"]:
            if r.vlm:
                st.text(f"Raw output (frame {r.frame_idx}):")
                st.code(r.vlm.raw_text, language="text")
                if r.vlm.parsed:
                    st.json(r.vlm.parsed)
                else:
                    st.warning("JSON parsing failed. Raw text shown above.")
            else:
                st.info("No VLM output for this frame (VLM skip or not enabled).")
                vlm_frames = [
                    res for res in results if res.vlm and (res.vlm.raw_text or res.vlm.parsed)
                ]
                if not vlm_frames:
                    st.warning("No VLM output exists in this analyzed run.")
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

        # ============================================================
        # TEMPORAL CHANGES TAB
        # ============================================================
        if "Temporal Changes" in tab_map:
            with tab_map["Temporal Changes"]:
                if r.temporal_changes:
                    tc = r.temporal_changes
                    st.markdown(f"**Comparing frames {tc.frame_idx_before} -> {tc.frame_idx_after}**")

                    if tc.state_changes:
                        st.markdown("**State Changes:**")
                        st.dataframe([
                            {
                                "Object": sc.object_name,
                                "Before": sc.before_state,
                                "After": sc.after_state,
                                "Confidence": sc.confidence,
                            }
                            for sc in tc.state_changes
                        ])
                    else:
                        st.info("No state changes detected between these frames.")

                    if tc.actions_detected:
                        st.markdown("**Actions Detected:**")
                        for action in tc.actions_detected:
                            st.write(f"- {action}")

                    with st.expander("Raw VLM Output"):
                        st.code(tc.raw_text, language="text")

                    vis = draw_temporal_changes(r.frame, tc)
                    st.image(cv2.cvtColor(vis, cv2.COLOR_BGR2RGB),
                             caption="Temporal Changes Overlay", width='stretch')
                else:
                    st.info("No temporal changes for this frame (requires VLM + temporal diff enabled).")

        # ============================================================
        # GROUND TRUTH JSON TAB
        # ============================================================
        with tab_map["Ground Truth JSON"]:
            st.markdown("**Structured Ground Truth for this frame:**")
            from pipeline.exporter import _frame_to_dict
            frame_gt = _frame_to_dict(r, info["fps"])
            st.json(frame_gt)

            st.divider()
            st.markdown("**Full Video Summary:**")
            gt = export_ground_truth(results, info, nav_timeline=nav_timeline)
            summary_cols = st.columns(3)
            summary_cols[0].metric("Frames Processed", gt["summary"]["total_frames_processed"])
            summary_cols[1].metric("Modalities", ", ".join(gt["summary"]["modalities"]))
            if "tracks_summary" in gt:
                summary_cols[2].metric("Unique Tracks", len(gt["tracks_summary"]))

            if "tracks_summary" in gt and gt["tracks_summary"]:
                st.markdown("**Track Summary:**")
                st.dataframe([
                    {
                        "Track ID": tid,
                        "Class": t["class"],
                        "First Frame": t["first_frame"],
                        "Last Frame": t["last_frame"],
                        "Frames Seen": t["frame_count"],
                        "Avg Confidence": f"{t['avg_confidence']:.3f}",
                    }
                    for tid, t in gt["tracks_summary"].items()
                ])

            if "navigation_ground_truth" in gt:
                st.markdown("**Navigation Ground Truth:**")
                st.json(gt["navigation_ground_truth"])

            st.divider()
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
                )
            with dl_col2:
                st.download_button(
                    "Download Analysis Results JSON",
                    analysis_json,
                    file_name="analysis_results.json",
                    mime="application/json",
                )

        # ============================================================
        # ACCURACY TAB
        # ============================================================
        with tab_map["Accuracy"]:
            st.markdown("### Accuracy Measurement")
            st.markdown(
                "Compare pipeline predictions against manual labels to measure ground truth quality. "
                "This addresses the evaluation criteria: **GT Accuracy** and **Zero-Shot vs Fine-Tuned Gains**."
            )

            if not nav_timeline or not nav_timeline.objects:
                st.warning("Run inference with Tracker + VLM (Navigation prompt) first to generate predictions.")
            else:
                st.markdown("#### Manual Labels")
                st.markdown(
                    "Enter manual ground truth labels for specific frames. "
                    "Format: one label per row — frame number, object name, true state."
                )

                # Option 1: Upload JSON
                uploaded_labels = st.file_uploader(
                    "Upload labels JSON",
                    type=["json"],
                    help='Format: [{"frame": 0, "object": "door", "state": "closed"}, ...]',
                )

                # Option 2: Manual entry
                with st.expander("Or enter labels manually"):
                    num_labels = st.number_input("Number of labels", 1, 50, 5, key="num_labels")
                    manual_entries = []
                    for i in range(int(num_labels)):
                        lcols = st.columns(3)
                        frame_num = lcols[0].number_input(f"Frame", 0, 100000, 0, key=f"lbl_frame_{i}")
                        obj_name = lcols[1].text_input(f"Object", value="door", key=f"lbl_obj_{i}")
                        obj_state = lcols[2].selectbox(
                            f"State", ["closed", "open", "ajar", "blocked", "clear", "unknown"],
                            key=f"lbl_state_{i}",
                        )
                        manual_entries.append({"frame": frame_num, "object": obj_name, "state": obj_state})

                eval_btn = st.button("Evaluate Accuracy", type="primary")

                if eval_btn:
                    # Parse labels
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

                        # Show results
                        st.markdown("#### Results")
                        rcols = st.columns(3)
                        rcols[0].metric("Overall Accuracy", f"{metrics['accuracy']:.1%}")
                        rcols[1].metric("Correct", f"{metrics['correct']} / {metrics['total_labels']}")
                        rcols[2].metric("Labels Evaluated", metrics["total_labels"])

                        # Per-state accuracy
                        if metrics["per_state"]:
                            st.markdown("**Per-State Accuracy:**")
                            st.dataframe([
                                {"State": state, "Correct": d["correct"],
                                 "Total": d["total"], "Accuracy": f"{d['accuracy']:.1%}"}
                                for state, d in metrics["per_state"].items()
                            ])

                        # Per-object accuracy
                        if metrics["per_object"]:
                            st.markdown("**Per-Object Accuracy:**")
                            st.dataframe([
                                {"Object": obj, "Correct": d["correct"],
                                 "Total": d["total"], "Accuracy": f"{d['accuracy']:.1%}"}
                                for obj, d in metrics["per_object"].items()
                            ])

                        # Confusion matrix
                        if metrics["confusion_matrix"]:
                            st.markdown("**Confusion Matrix:**")
                            st.dataframe([
                                {"True -> Predicted": k, "Count": v}
                                for k, v in metrics["confusion_matrix"].items()
                            ])

                        # Detail table
                        with st.expander("Detailed Results"):
                            st.dataframe(metrics["details"])

        # --- Export Video ---
        st.divider()
        export_col1, export_col2 = st.columns([1, 3])
        with export_col1:
            export_btn = st.button("Export Annotated Video")

        if export_btn:
            with st.spinner("Rendering video..."):
                out_path = Path(tempfile.mktemp(suffix=".mp4"))
                export_annotated_video(results, out_path, info["fps"] / frame_skip, draw_all)
                video_bytes = out_path.read_bytes()
                st.session_state["exported_video"] = video_bytes
                out_path.unlink(missing_ok=True)

        # Show video player + download if export exists
        if st.session_state["exported_video"] is not None:
            st.video(st.session_state["exported_video"], format="video/mp4")
            st.download_button(
                "Download MP4",
                st.session_state["exported_video"],
                file_name="annotated_output.mp4",
                mime="video/mp4",
            )
