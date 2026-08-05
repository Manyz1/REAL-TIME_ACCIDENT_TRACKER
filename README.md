<div align="center">
  
# 🚨 Real-Time Accident Tracker
**AI-Powered Vehicle Accident Detection & Verification System**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://real-timeaccidenttracker-j6ebp9myxsylcqvtjcfepr.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLO-v8-yellow.svg)](https://github.com/ultralytics/ultralytics)
[![OpenAI CLIP](https://img.shields.io/badge/OpenAI-CLIP-green.svg)](https://github.com/openai/CLIP)

An advanced, real-time computer vision pipeline that detects traffic accidents by combining **YOLOv8** object tracking (speed-drop heuristics) with **OpenAI CLIP** (semantic visual verification) to eliminate false positives.

</div>

---

## 🌐 Live Demo (Cloud Mode)

You can try the system without installing anything locally via our Streamlit Cloud deployment:

👉 **[Launch Accident Tracker on Streamlit](https://real-timeaccidenttracker-j6ebp9myxsylcqvtjcfepr.streamlit.app/)**

**Cloud Limitations:**
*   Accepts pre-recorded video uploads only (No live screen capture).
*   Runs on CPU using lightweight models (`yolov8n.pt` & `CLIP-ViT-Base`).
*   Processing takes a few minutes depending on the video length.

---

## 💻 Local Setup (Full Features)

For real-time desktop window capture, live display, GPU acceleration, and maximum accuracy, run the system locally on a Windows machine.

### Prerequisites
*   Windows 10 / 11
*   Python 3.10 or newer
*   NVIDIA GPU (Highly recommended for real-time performance)

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Manyz1/REAL-TIME_ACCIDENT_TRACKER.git
   cd REAL-TIME_ACCIDENT_TRACKER
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements_local.txt
   ```

3. **Run the Dashboard:**
   ```bash
   python -m streamlit run app.py
   ```
   *The app will automatically open in your browser at `http://localhost:8501`.*

---

## ⚙️ How It Works (The Pipeline)

The system uses a robust two-stage verification architecture to ensure high accuracy and low false-alarm rates:

1.  **Detection & Tracking:** **YOLOv8 + ByteTrack** detects and tracks vehicles (cars, trucks, buses) frame-by-frame.
2.  **Kinematic Analysis:** The system computes pixel displacement per frame to monitor the relative speed of every tracked vehicle.
3.  **Heuristic Trigger:** If a vehicle's speed drops abruptly below a set ratio (e.g., `< 20%` of its peak speed), it flags a **Suspected Crash**.
4.  **Semantic Verification:** The exact frames of the suspected crash are sent to a background thread running **OpenAI CLIP**. CLIP compares the visual scene against prompts like *"a severe car accident"* vs *"normal traffic"*.
5.  **Confirmation & Alert:** If CLIP confirms the accident, the UI triggers a permanent alert, saves the highest-scoring evidence frame to the `crashes/` folder, and visually marks the output video.

---

## 🎛️ Configuration Parameters

You can tweak the pipeline in real-time using the Streamlit sidebar:

| Parameter | Default | Description |
| :--- | :--- | :--- |
| **Confidence** | `0.40` | Minimum YOLO score for a detection to be accepted. |
| **IoU** | `0.50` | Overlap suppression to reduce duplicate detections. |
| **Resolution** | `416` (Cloud) | Input frame size for YOLO. Higher = better accuracy but slower. |
| **Analysis Interval** | `1.0s` | How frequently the speed & crash heuristics run. |
| **Speed Drop Ratio** | `0.20` | Fraction of peak speed that triggers a sudden stop alert. |
| **Min Peak Speed** | `5 px/f` | Ignores parked or slow-moving vehicles from tracking. |
| **CLIP Threshold** | `0.85` | Score required for the AI to definitively confirm a crash. |

---

## 📁 Project Structure

```text
├── app.py                  # Main Streamlit dashboard (Auto-detects Cloud vs Local)
├── pipeline.py             # Local pipeline implementation (Window Capture APIs)
├── pipeline_cloud.py       # Cloud pipeline implementation (Video Processing only)
├── clip_verifier.py        # Background thread handling CLIP inference
├── analyzer.py             # Speed computation and crash heuristic logic
├── dashboard.py            # OpenCV UI drawing (Danger bar, speeds, status)
├── config.py               # Hyperparameters and model defaults
├── requirements.txt        # Dependencies for Streamlit Cloud (Linux)
├── requirements_local.txt  # Dependencies for Local Windows execution
├── packages.txt            # System-level dependencies for Streamlit servers (libgl1)
└── crashes/                # Directory where confirmed accident frames are saved
```

---
*Developed by [Manyz1](https://github.com/Manyz1).*
