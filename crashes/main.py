import time
import datetime
import cv2
import numpy as np
import torch
from ctypes import windll
from collections import defaultdict, deque
from ultralytics import YOLO
import win32gui
import win32ui

from config import (YOLO_MODEL, YOLO_CONF, YOLO_IOU, YOLO_CLASSES,
                    YOLO_IMGSZ, MAX_FRAMES, ANALYZER_INTERVAL, CLIP_THRESHOLD)
from analyzer import compute_pixel_speed, smooth_speed, detect_crashes, get_crash_frames
from clip_verifier import crash_queue, danger_level, last_accident_ts, load_clip, start_clip_thread
from dashboard import draw_dashboard


def list_windows():
    windows = []
    def cb(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append((hwnd, title))
    win32gui.EnumWindows(cb, None)
    return windows


def capture_window(hwnd):
    try:
        left, top, right, bot = win32gui.GetClientRect(hwnd)
        w = right - left
        h = bot - top
        if w == 0 or h == 0:
            return None

        hwnd_dc  = win32gui.GetWindowDC(hwnd)
        mfc_dc   = win32ui.CreateDCFromHandle(hwnd_dc)
        save_dc  = mfc_dc.CreateCompatibleDC()
        bmp      = win32ui.CreateBitmap()
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


def main():
    windows = list_windows()
    for i, (_, title) in enumerate(windows):
        print(f"  {i:3}: {title}")

    idx        = int(input("\nاختار رقم النافذة: "))
    hwnd, title = windows[idx]
    print(f"\nCapturing: {title}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading YOLO... (device: {device})")
    model = YOLO(YOLO_MODEL)
    model.to(device)

    print("Loading CLIP...")
    load_clip()
    start_clip_thread()
    print("All models ready. Press Q to quit.\n")

    frame_data     = defaultdict(lambda: deque(maxlen=MAX_FRAMES))
    raw_frames     = deque(maxlen=MAX_FRAMES)
    pixel_speed    = {}
    frame_idx      = 0
    last_tick      = time.time()
    checked_tracks = {}

    while True:
        frame = capture_window(hwnd)
        if frame is None:
            if time.time() - last_tick >= ANALYZER_INTERVAL:
                danger_level[0] = max(0.0, danger_level[0] - 0.1)
                last_tick = time.time()
            time.sleep(0.05)
            continue

        results = model.track(
            frame,
            conf=YOLO_CONF,
            iou=YOLO_IOU,
            classes=YOLO_CLASSES,
            persist=True,
            tracker="bytetrack.yaml",
            imgsz=YOLO_IMGSZ,
            verbose=False,
        )[0]

        raw_frames.append((frame_idx, frame.copy()))

        if results.boxes is not None and results.boxes.id is not None:
            ids   = results.boxes.id.cpu().numpy().astype(int)
            boxes = results.boxes.xyxy.cpu().numpy()
            for tid, box in zip(ids, boxes):
                x1, y1, x2, y2 = box
                frame_data[tid].append((frame_idx, (x1 + x2) / 2, (y1 + y2) / 2))

        if time.time() - last_tick >= ANALYZER_INTERVAL:
            pixel_speed = compute_pixel_speed(frame_data)
            suspected   = detect_crashes(pixel_speed)

            new_suspected = []
            for tid in suspected:
                if tid not in checked_tracks:
                    new_suspected.append(tid)
                else:
                    entries = list(frame_data[tid])
                    if entries and entries[0][0] > checked_tracks[tid]:
                        new_suspected.append(tid)

            if new_suspected:
                danger_level[0] = max(danger_level[0], 0.75)
                crash_frames = get_crash_frames(new_suspected, pixel_speed, frame_data, raw_frames)
                if crash_frames:
                    ts = datetime.datetime.now().strftime("%H%M%S")
                    crash_queue.put((crash_frames, ts, CLIP_THRESHOLD))
                for tid in new_suspected:
                    entries = list(frame_data[tid])
                    checked_tracks[tid] = entries[0][0] if entries else frame_idx
            else:
                danger_level[0] = max(0.0, danger_level[0] - 0.1)
                checked_tracks = {tid: f for tid, f in checked_tracks.items() if tid in frame_data}

            last_tick = time.time()

        annotated = draw_dashboard(results.plot(), pixel_speed, danger_level[0], last_accident_ts[0])
        cv2.imshow("Real-Time Accident Tracker", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_idx += 1

    crash_queue.put(None)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
