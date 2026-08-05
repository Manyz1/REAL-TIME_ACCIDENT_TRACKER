import os
import base64
import tempfile
import datetime
import streamlit as st
import clip_verifier
from config import (
    YOLO_CONF, YOLO_IOU, YOLO_IMGSZ,
    CLOUD_YOLO_IMGSZ,
    ANALYZER_INTERVAL, DROP_RATIO, MIN_PEAK_SPEED, CLIP_THRESHOLD,
)

# ── Environment detection ────────────────────────────────────────────────────
try:
    import win32gui
    _IS_LOCAL = True
except ImportError:
    _IS_LOCAL = False

if _IS_LOCAL:
    from pipeline import pipeline, list_windows
else:
    from pipeline_cloud import cloud_pipeline as pipeline

ICONS_DIR = "assets/icons"

_app_icon = os.path.join(ICONS_DIR, "app.png")
st.set_page_config(
    page_title="Accident Tracker",
    page_icon=_app_icon if os.path.isfile(_app_icon) else "🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _icon(name: str, size: int = 20) -> str:
    path = os.path.join(ICONS_DIR, f"{name}.png")
    if not os.path.isfile(path):
        return ""
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    return (
        f'<img src="data:image/png;base64,{b64}" '
        f'width="{size}" height="{size}" '
        f'style="vertical-align:middle;margin-right:7px;opacity:0.9;">'
    )


st.markdown("""
<style>
    [data-testid="stSidebar"] { background: #111318; }
    [data-testid="stSidebar"] .stMarkdown h4 {
        color: #9ca3af; font-size: 0.7rem; font-weight: 600;
        text-transform: uppercase; letter-spacing: 1.2px; margin: 18px 0 6px;
    }
    [data-testid="stSidebar"] .stMarkdown p { color: #6b7280; font-size: 0.75rem; margin: 0; }

    .mode-badge {
        display: inline-block; padding: 3px 12px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .badge-cloud { background: #1e3a5f; color: #60a5fa; border: 1px solid #1d4ed8; }
    .badge-local { background: #064e3b; color: #34d399; border: 1px solid #065f46; }

    .status-card { border-radius: 10px; padding: 20px 22px; text-align: center; margin-bottom: 8px; }
    .s-idle      { background: #1f2937; border: 1px solid #374151; }
    .s-normal    { background: #064e3b; border: 1px solid #065f46; }
    .s-suspected { background: #78350f; border: 1px solid #92400e; }
    .s-confirmed { background: #7f1d1d; border: 1px solid #991b1b; }
    .s-done      { background: #1e3a5f; border: 1px solid #1d4ed8; }
    .status-title { font-size: 1.05rem; font-weight: 700; color: #f3f4f6; display: block; margin-bottom: 3px; }
    .status-sub   { font-size: 0.78rem; color: #d1d5db; }

    .danger-bar-wrap { background: #1f2937; border-radius: 6px; height: 10px; overflow: hidden; margin: 6px 0 2px; }
    .danger-bar-fill { height: 100%; border-radius: 6px; transition: width 0.4s ease, background 0.4s ease; }

    .track-row {
        display: flex; justify-content: space-between; align-items: center;
        background: #1a1d24; border-radius: 6px; padding: 6px 12px; margin: 3px 0;
        font-family: monospace; font-size: 0.82rem;
    }

    .info-box {
        background: #1e2433; border: 1px solid #2a3a5c; border-radius: 10px;
        padding: 16px 20px; margin: 10px 0;
    }
    .info-box code {
        background: #0f1117; padding: 2px 6px; border-radius: 4px;
        color: #93c5fd; font-size: 0.85rem;
    }

    div[data-testid="stButton"] > button {
        border-radius: 8px; font-weight: 600; width: 100%;
        font-size: 0.9rem; padding: 8px 0; border: 1px solid #2a2d36;
    }
    .section-label {
        font-size: 0.7rem; font-weight: 600; color: #6b7280;
        text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px;
    }
    hr { border-color: #1f2937; margin: 16px 0; }
</style>
""", unsafe_allow_html=True)


# ── SIDEBAR ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Accident Tracker")

    if _IS_LOCAL:
        st.markdown('<span class="mode-badge badge-local">Local Mode</span>', unsafe_allow_html=True)
        st.caption("Running on your machine — all features available.")
    else:
        st.markdown('<span class="mode-badge badge-cloud">Cloud Mode</span>', unsafe_allow_html=True)
        st.caption("Running on Streamlit Cloud — video processing only.")

    st.markdown("#### Detection")
    yolo_conf = st.slider("Confidence threshold", 0.10, 1.00, YOLO_CONF, 0.05)
    st.caption("Minimum score for a detection to be accepted. Lower values catch more objects but risk more false positives.")

    yolo_iou = st.slider("IoU threshold", 0.10, 1.00, YOLO_IOU, 0.05)
    st.caption("Controls how aggressively overlapping boxes are merged. Lower values reduce duplicate detections on the same vehicle.")

    _default_imgsz = YOLO_IMGSZ if _IS_LOCAL else CLOUD_YOLO_IMGSZ
    yolo_imgsz = st.select_slider("Input resolution", [320, 416, 512, 640, 1280], _default_imgsz)
    st.caption("Frame size sent to YOLO. Higher resolution improves accuracy on small or distant vehicles but increases processing time.")

    st.markdown("#### Analysis")
    analyzer_interval = st.slider("Analysis interval (s)", 0.5, 5.0, ANALYZER_INTERVAL, 0.5)
    st.caption("How often the speed and crash analysis runs. A lower value gives faster alerts but uses more CPU.")

    drop_ratio = st.slider("Speed drop ratio", 0.05, 0.50, DROP_RATIO, 0.05)
    st.caption("A vehicle is flagged when its current speed falls below this fraction of its peak speed — indicating a sudden stop.")

    min_peak_speed = st.slider("Min peak speed (px/frame)", 1, 30, MIN_PEAK_SPEED, 1)
    st.caption("Tracks that never exceed this speed are ignored. Prevents stationary or slow-moving parked cars from triggering alerts.")

    st.markdown("#### CLIP Verifier")
    clip_threshold = st.slider("Confirmation threshold", 0.50, 1.00, CLIP_THRESHOLD, 0.05)
    st.caption("Minimum CLIP score to confirm an accident. Higher values mean stricter verification — fewer false positives but may miss some events.")

    if not _IS_LOCAL:
        st.divider()
        with st.expander("Need Live Capture? — Run Locally"):
            st.markdown("""
**To use window capture and live display, run the app on your own Windows machine:**

**1. Clone the repository**
```
git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
cd YOUR_REPO
```

**2. Install dependencies**
```
pip install -r requirements_local.txt
```

**3. Run the app**
```
python -m streamlit run app.py
```

Then open **http://localhost:8501** in your browser.

> Windows 10/11 required. A GPU is recommended for real-time performance.
""")


# ── MAIN ─────────────────────────────────────────────────────────────────────

st.markdown("## Accident Tracker")
if _IS_LOCAL:
    st.caption("Local mode — window capture, live display, and full model quality available.")
else:
    st.caption("Cloud mode — upload a video file and download the annotated result.")
st.markdown("<hr>", unsafe_allow_html=True)


# ── CLOUD UI ─────────────────────────────────────────────────────────────────

if not _IS_LOCAL:
    st.markdown('<p class="section-label">Upload Video</p>', unsafe_allow_html=True)

    uploaded = st.file_uploader(
        "video_upload",
        type=["mp4", "avi", "mov", "mkv"],
        label_visibility="collapsed",
    )
    st.caption("Upload a dashcam or traffic video. The model will detect vehicles, track speeds, and flag potential accidents.")

    tmp_video_path  = None
    tmp_output_path = None

    if uploaded is not None:
        tmp_input  = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_input.write(uploaded.read())
        tmp_input.close()
        tmp_video_path = tmp_input.name

        tmp_output      = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tmp_output.close()
        tmp_output_path = tmp_output.name

    st.markdown("<hr>", unsafe_allow_html=True)

    btn_a, btn_b, btn_c = st.columns(3, gap="small")
    with btn_a:
        can_start = (tmp_video_path is not None) and not pipeline.running
        if st.button("Process Video", disabled=not can_start):
            cfg = {
                "video_path":        tmp_video_path,
                "output_path":       tmp_output_path,
                "yolo_model":        "yolov8n.pt",
                "yolo_conf":         yolo_conf,
                "yolo_iou":          yolo_iou,
                "yolo_imgsz":        yolo_imgsz,
                "analyzer_interval": analyzer_interval,
                "drop_ratio":        drop_ratio,
                "min_peak_speed":    min_peak_speed,
                "clip_threshold":    clip_threshold,
            }
            pipeline.start(cfg)
            st.rerun()

    with btn_b:
        if st.button("Cancel", disabled=not pipeline.running):
            pipeline.stop()
            st.rerun()

    with btn_c:
        if st.button("Reset Status", disabled=pipeline.running):
            pipeline.reset_status()
            st.rerun()

    st.caption("Process Video runs the full pipeline on your uploaded file. This may take a few minutes depending on video length.")


# ── LOCAL UI ─────────────────────────────────────────────────────────────────

else:
    src_col, out_col = st.columns(2, gap="large")

    with src_col:
        st.markdown('<p class="section-label">Input Source</p>', unsafe_allow_html=True)
        source_type = st.radio(
            "source", ["Live Window", "Video File"],
            horizontal=True, label_visibility="collapsed",
        )

        hwnd       = None
        video_path = None

        if source_type == "Live Window":
            st.caption("Grabs frames in real time from any open window on your desktop. Useful for monitoring dashcam software, live feeds, or simulators.")
            ref_col, sel_col = st.columns([1, 6])
            with ref_col:
                if st.button("Refresh", help="Reload the list of open windows"):
                    st.session_state.pop("windows_list", None)
                    st.rerun()
            with sel_col:
                if "windows_list" not in st.session_state:
                    st.session_state.windows_list = list_windows()
                windows      = st.session_state.windows_list
                window_names = [t for _, t in windows]
                selected     = st.selectbox("Window", window_names, label_visibility="collapsed")
                hwnd         = windows[window_names.index(selected)][0] if selected else None
        else:
            st.caption("Process a recorded video file from disk. The model reads every frame at the file's native frame rate. Supported formats: MP4, AVI, MOV.")
            video_path = st.text_input("File path", placeholder=r"C:\videos\dashcam.mp4", label_visibility="collapsed")
            if video_path and not os.path.isfile(video_path):
                st.error("File not found — please check the path.")
                video_path = None

    with out_col:
        st.markdown('<p class="section-label">Output Mode</p>', unsafe_allow_html=True)
        output_mode = st.radio(
            "output", ["Live Display", "Save as Video"],
            horizontal=True, label_visibility="collapsed",
        )
        output_path = None

        if output_mode == "Live Display":
            st.caption("Opens a separate OpenCV window showing the annotated feed in real time. Bounding boxes, track IDs, speed, and the danger bar are rendered there.")
        else:
            st.caption("Encodes every annotated frame and writes the result to an MP4 file. No live window is opened — ideal for batch-processing recorded footage.")
            default_name = f"output_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
            output_path  = st.text_input("Save to", value=default_name, label_visibility="collapsed")
            st.caption("Full file path including filename. Leave as default to save in the project folder.")
            if output_path:
                out_dir = os.path.dirname(output_path) or "."
                if not os.path.isdir(out_dir):
                    st.error("Output directory does not exist.")
                    output_path = None

    st.markdown("<hr>", unsafe_allow_html=True)

    source_ready = (hwnd is not None) or (video_path is not None)
    output_ready = (output_mode == "Live Display") or (output_path is not None)
    can_start    = source_ready and output_ready and not pipeline.running

    btn_a, btn_b, btn_c = st.columns(3, gap="small")
    with btn_a:
        if st.button("Start", disabled=not can_start):
            cfg = {
                "source_type":       "window" if source_type == "Live Window" else "video",
                "hwnd":              hwnd,
                "video_path":        video_path,
                "output_mode":       "live" if output_mode == "Live Display" else "save",
                "output_path":       output_path,
                "yolo_conf":         yolo_conf,
                "yolo_iou":          yolo_iou,
                "yolo_imgsz":        yolo_imgsz,
                "analyzer_interval": analyzer_interval,
                "drop_ratio":        drop_ratio,
                "min_peak_speed":    min_peak_speed,
                "clip_threshold":    clip_threshold,
            }
            pipeline.start(cfg)
            st.rerun()

    with btn_b:
        if st.button("Stop", disabled=not pipeline.running):
            pipeline.stop()
            st.rerun()

    with btn_c:
        if st.button("Reset Status", disabled=pipeline.running):
            pipeline.reset_status()
            st.rerun()

    st.caption("Start begins the pipeline with the settings above. Stop halts it immediately. Reset Status clears a confirmed accident alert.")


# ── STATUS PANEL (shared) ────────────────────────────────────────────────────

st.markdown("<hr>", unsafe_allow_html=True)


@st.fragment(run_every=1.0)
def status_panel():
    snap       = pipeline.state
    danger     = snap["danger_level"]
    ts         = snap["last_accident_ts"]
    speeds     = snap["pixel_speed"]
    running    = snap["running"]
    progress   = snap["progress"]
    video_done = snap["video_done"]
    out_path   = snap.get("output_path")
    err_msg    = snap.get("error_msg")

    left, right = st.columns([1, 1], gap="large")

    if err_msg:
        st.error(f"Pipeline crashed: {err_msg}")

    with left:
        st.markdown('<p class="section-label">Detection Status</p>', unsafe_allow_html=True)

        if video_done and not running:
            css, icon_n, title, sub = "s-done",      "done",      "Video Processed",    "The file has been fully analysed."
        elif not running and not ts:
            css, icon_n, title, sub = "s-idle",      "idle",      "Idle",               "Press Start to begin."
        elif ts:
            css, icon_n, title, sub = "s-confirmed", "confirmed", "Accident Confirmed", f"Detected at {ts}"
        elif danger >= 0.7:
            css, icon_n, title, sub = "s-suspected", "suspected", "Suspected Crash",    "Sending frames to CLIP for verification…"
        else:
            css, icon_n, title, sub = "s-normal",    "normal",    "Normal Traffic",     "No incidents detected."

        st.markdown(
            f'<div class="status-card {css}">'
            f'{_icon(icon_n, size=22)}'
            f'<span class="status-title">{title}</span>'
            f'<span class="status-sub">{sub}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

        bar_color = "#ef4444" if danger >= 0.7 else ("#f59e0b" if danger >= 0.4 else "#10b981")
        st.markdown(
            f'<p style="font-size:0.78rem;color:#6b7280;margin:10px 0 2px;">Danger Level — {danger*100:.0f}%</p>'
            f'<div class="danger-bar-wrap">'
            f'<div class="danger-bar-fill" style="width:{danger*100:.0f}%;background:{bar_color};"></div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        if progress > 0:
            st.write("")
            st.markdown(f'<p style="font-size:0.78rem;color:#6b7280;margin:10px 0 4px;">Video Progress — {progress*100:.1f}%</p>', unsafe_allow_html=True)
            st.progress(progress)

        if not _IS_LOCAL and video_done and out_path and os.path.isfile(out_path):
            st.write("")
            with open(out_path, "rb") as f:
                st.download_button(
                    label="Download Annotated Video",
                    data=f,
                    file_name="accident_tracker_output.mp4",
                    mime="video/mp4",
                )
            st.caption("The output video contains bounding boxes, track IDs, speeds, and the danger bar overlay.")

    with right:
        st.markdown('<p class="section-label">Tracked Vehicles</p>', unsafe_allow_html=True)
        if speeds:
            for tid, spd_dict in list(speeds.items())[:10]:
                if not spd_dict:
                    continue
                spd     = list(spd_dict.values())[-1]
                spd_cls = "color:#f87171;" if spd < 1.5 else "color:#93c5fd;"
                fill    = min(int(spd / 3), 10)
                bar     = "▮" * fill + "▯" * (10 - fill)
                st.markdown(
                    f'<div class="track-row">'
                    f'<span style="color:#6b7280;">Track {tid:>3}</span>'
                    f'<span style="{spd_cls}font-weight:600;">{spd:>6.1f} px/f</span>'
                    f'<span style="color:#374151;font-size:0.75rem;">{bar}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.caption("No vehicles are being tracked yet.")


status_panel()
