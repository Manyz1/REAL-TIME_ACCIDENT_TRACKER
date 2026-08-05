import time
import datetime
import threading
import cv2
import numpy as np
import torch
from ctypes import windll
from collections import defaultdict, deque
from ultralytics import YOLO
import win32gui
import win32ui

from config import YOLO_MODEL, MAX_FRAMES
from analyzer import compute_pixel_speed, detect_crashes, get_crash_frames
from dashboard import draw_dashboard
import clip_verifier


def list_windows():
    windows = []
    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append((hwnd, title))
    win32gui.EnumWindows(cb, None)
    return windows


def _capture_window(hwnd):
    try:
        left, top, right, bot = win32gui.GetClientRect(hwnd)
        w, h = right - left, bot - top
        if w == 0 or h == 0:
            return None
        hwnd_dc = win32gui.GetWindowDC(hwnd)
        mfc_dc  = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc = mfc_dc.CreateCompatibleDC()
        bmp     = win32ui.CreateBitmap()
        bmp.CreateCompatibleBitmap(mfc_dc, w, h)
        save_dc.SelectObject(bmp)
        windll.user32.PrintWindow(hwnd, save_dc.GetSafeHdc(), 3)
        bmp_bits = bmp.GetBitmapBits(True)
        img = np.frombuffer(bmp_bits, dtype=np.uint8).reshape(h, w, 4)
        win32gui.DeleteObject(bmp.GetHandle())
        save_dc.DeleteDC()
        mfc_dc.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwnd_dc)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
    except Exception:
        return None


class Pipeline:
    def __init__(self):
        self._lock         = threading.Lock()
        self._thread       = None
        self._stop_event   = threading.Event()
        self._pixel_speed  = {}
        self._progress     = 0.0
        self._video_done   = False
        self._clip_ready   = False
        self.running       = False

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
        return {
            'pixel_speed':      speeds,
            'danger_level':     clip_verifier.danger_level[0],
            'last_accident_ts': clip_verifier.last_accident_ts[0],
            'running':          self.running,
            'progress':         progress,
            'video_done':       video_done,
        }

    def reset_status(self):
        clip_verifier.danger_level[0]     = 0.0
        clip_verifier.last_accident_ts[0] = None

    def start(self, cfg):
        if self.running:
            return
        self._ensure_clip()
        self.reset_status()
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
        source_type = cfg['source_type']   # 'window' | 'video'
        output_mode = cfg['output_mode']   # 'live'   | 'save'

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model  = YOLO(YOLO_MODEL)
        model.to(device)

        cap          = None
        source_fps   = 30.0
        total_frames = 0

        if source_type == 'video':
            cap          = cv2.VideoCapture(cfg['video_path'])
            source_fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_interval = 1.0 / source_fps

        frame_data     = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
        raw_frames     = deque(maxlen=MAX_FRAMES)
        pixel_speed    = {}
        frame_idx      = 0
        last_tick      = time.time()
        checked_tracks = {}
        writer         = None

        while not self._stop_event.is_set():
            t_loop = time.time()

            if source_type == 'video':
                ret, frame = cap.read()
                if not ret:
                    with self._lock:
                        self._video_done = True
                    break
            else:
                frame = _capture_window(cfg['hwnd'])
                if frame is None:
                    if time.time() - last_tick >= cfg['analyzer_interval']:
                        clip_verifier.danger_level[0] = max(0.0, clip_verifier.danger_level[0] - 0.1)
                        last_tick = time.time()
                    time.sleep(0.05)
                    continue

            if output_mode == 'save' and writer is None:
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
                imgsz=cfg['yolo_imgsz'],
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

            if output_mode == 'live':
                cv2.imshow("Real-Time Accident Tracker", annotated)
                elapsed  = time.time() - t_loop
                wait_ms  = max(1, int((frame_interval - elapsed) * 1000))
                cv2.waitKey(wait_ms)
            else:
                if writer is not None:
                    writer.write(annotated)

            frame_idx += 1

        if cap:
            cap.release()
        if writer:
            writer.release()
        if output_mode == 'live':
            cv2.destroyAllWindows()

        self.running = False


pipeline = Pipeline()
