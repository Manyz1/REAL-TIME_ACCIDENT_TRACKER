import os
import time
import datetime
import threading
import cv2
import numpy as np
import torch
from collections import defaultdict, deque
from ultralytics import YOLO

from config import CLOUD_YOLO_MODEL, CLOUD_YOLO_IMGSZ, MAX_FRAMES
from analyzer import compute_pixel_speed, detect_crashes, get_crash_frames
from dashboard import draw_dashboard
import clip_verifier


class CloudPipeline:
    def __init__(self):
        self._lock        = threading.Lock()
        self._thread      = None
        self._stop_event  = threading.Event()
        self._pixel_speed = {}
        self._progress    = 0.0
        self._video_done  = False
        self._clip_ready  = False
        self.running      = False
        self.output_path  = None
        self.error_msg    = None

    def _ensure_clip(self):
        if not self._clip_ready:
            clip_verifier.load_clip()
            clip_verifier.start_clip_thread()
            self._clip_ready = True

    @property
    def state(self):
        with self._lock:
            speeds     = dict(self._pixel_speed)
            progress   = self._progress
            video_done = self._video_done
            err        = self.error_msg
        return {
            'pixel_speed':      speeds,
            'danger_level':     clip_verifier.danger_level[0],
            'last_accident_ts': clip_verifier.last_accident_ts[0],
            'running':          self.running,
            'progress':         progress,
            'video_done':       video_done,
            'output_path':      self.output_path,
            'error_msg':        err,
        }

    def reset_status(self):
        clip_verifier.danger_level[0]     = 0.0
        clip_verifier.last_accident_ts[0] = None

    def start(self, cfg):
        if self.running:
            return
        self._ensure_clip()
        self.reset_status()
        self.output_path = cfg['output_path']
        self.error_msg   = None
        with self._lock:
            self._pixel_speed = {}
            self._progress    = 0.0
            self._video_done  = False
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, args=(cfg,), daemon=True)
        self._thread.start()
        self.running = True

    def stop(self):
        self._stop_event.set()
        self.running = False
        self.reset_status()

    def _run(self, cfg):
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model  = YOLO(cfg.get('yolo_model', CLOUD_YOLO_MODEL))
            model.to(device)

            cap          = cv2.VideoCapture(cfg['video_path'])
            source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_data     = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
            raw_frames     = deque(maxlen=MAX_FRAMES)
            pixel_speed    = {}
            frame_idx      = 0
            last_tick      = time.time()
            checked_tracks = {}
            writer         = None

            while not self._stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    with self._lock:
                        self._video_done = True
                    break

                if writer is None:
                    h, w   = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(cfg['output_path'], fourcc, source_fps, (w, h))

                results = model.track(
                    frame,
                    conf=cfg['yolo_conf'],
                    iou=cfg['yolo_iou'],
                    classes=[2, 3, 5, 7],
                    persist=True,
                    tracker="bytetrack.yaml",
                    imgsz=cfg.get('yolo_imgsz', CLOUD_YOLO_IMGSZ),
                    verbose=False,
                )[0]

                raw_frames.append((frame_idx, frame.copy()))

                if results.boxes is not None and results.boxes.id is not None:
                    ids   = results.boxes.id.cpu().numpy().astype(int)
                    boxes = results.boxes.xyxy.cpu().numpy()
                    for tid, box in zip(ids, boxes):
                        x1, y1, x2, y2 = box
                        frame_data[tid].append((frame_idx, (x1 + x2) / 2, (y1 + y2) / 2))

                if time.time() - last_tick >= cfg['analyzer_interval']:
                    pixel_speed = compute_pixel_speed(frame_data)
                    suspected   = detect_crashes(
                        pixel_speed,
                        min_peak=cfg['min_peak_speed'],
                        drop_ratio=cfg['drop_ratio'],
                    )

                    new_suspected = []
                    for tid in suspected:
                        if tid not in checked_tracks:
                            new_suspected.append(tid)
                        else:
                            entries = list(frame_data[tid])
                            if entries and entries[0][0] > checked_tracks[tid]:
                                new_suspected.append(tid)

                    if new_suspected:
                        clip_verifier.danger_level[0] = max(clip_verifier.danger_level[0], 0.75)
                        crash_frames = get_crash_frames(new_suspected, pixel_speed, frame_data, raw_frames)
                        if crash_frames:
                            ts = datetime.datetime.now().strftime("%H%M%S")
                            clip_verifier.crash_queue.put((crash_frames, ts, cfg['clip_threshold']))
                        for tid in new_suspected:
                            entries = list(frame_data[tid])
                            checked_tracks[tid] = entries[0][0] if entries else frame_idx
                    else:
                        clip_verifier.danger_level[0] = max(0.0, clip_verifier.danger_level[0] - 0.1)
                        checked_tracks = {tid: f for tid, f in checked_tracks.items() if tid in frame_data}

                    last_tick = time.time()

                with self._lock:
                    self._pixel_speed = pixel_speed
                    if total_frames > 0:
                        self._progress = min(frame_idx / total_frames, 1.0)

                annotated = draw_dashboard(
                    results.plot(),
                    pixel_speed,
                    clip_verifier.danger_level[0],
                    clip_verifier.last_accident_ts[0],
                )

                if writer is not None:
                    writer.write(annotated)

                frame_idx += 1

        except Exception as e:
            self.error_msg = f"Error in pipeline: {str(e)}"
            print(f"PIPELINE ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if 'cap' in locals(): cap.release()
            if 'writer' in locals() and writer is not None: writer.release()
            self.running = False


cloud_pipeline = CloudPipeline()
