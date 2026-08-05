import math
import numpy as np
from scipy.signal import medfilt
from config import MIN_PEAK_SPEED, DROP_RATIO, STOP_RATIO, STOP_THRESHOLD, MEDFILT_KERNEL, CRASH_FRAME_COUNT


def compute_pixel_speed(frame_data):
    pixel_speed = {}
    for tid, entries in frame_data.items():
        entries = list(entries)
        if len(entries) < 2:
            continue
        pixel_speed[tid] = {}
        for i in range(1, len(entries)):
            f1, x1, y1 = entries[i - 1]
            f2, x2, y2 = entries[i]
            dist = math.hypot(x2 - x1, y2 - y1)
            pixel_speed[tid][f2] = dist / max(f2 - f1, 1)
    return pixel_speed


def smooth_speed(pixel_speed):
    filtered = {}
    for tid, speeds in pixel_speed.items():
        frames = list(speeds.keys())
        vals   = list(speeds.values())
        k = MEDFILT_KERNEL if len(vals) >= MEDFILT_KERNEL else (len(vals) if len(vals) % 2 == 1 else max(len(vals) - 1, 1))
        filtered[tid] = dict(zip(frames, medfilt(vals, kernel_size=k)))
    return filtered


def detect_crashes(pixel_speed, min_peak=None, drop_ratio=None, stop_ratio=None):
    _min_peak  = min_peak   if min_peak   is not None else MIN_PEAK_SPEED
    _drop      = drop_ratio if drop_ratio is not None else DROP_RATIO
    _stop      = stop_ratio if stop_ratio is not None else STOP_RATIO

    suspected = []
    for tid, speeds in pixel_speed.items():
        if len(speeds) < 5:
            continue
        vals = list(speeds.values())
        peak = max(vals)
        if peak < _min_peak:
            continue
        final_speed = vals[-1]
        drop = final_speed / peak
        if drop <= _drop and final_speed <= peak * _stop:
            suspected.append(tid)
    return suspected


def get_crash_frames(suspected, pixel_speed, frame_data, raw_frames,
                      n_frames=CRASH_FRAME_COUNT, stop_threshold=STOP_THRESHOLD):
    raw_dict = {idx: img for idx, img in raw_frames}
    all_frames = []

    for tid in suspected:
        speeds = pixel_speed.get(tid, {})
        if not speeds:
            continue

        stop_frame = None
        for fid, spd in sorted(speeds.items()):
            if spd <= stop_threshold:
                stop_frame = fid
                break

        if stop_frame is None:
            stop_frame = min(speeds, key=speeds.get)

        entries = frame_data.get(tid)
        if not entries:
            continue
        last_frame = max(f for f, _, _ in entries)

        if last_frame <= stop_frame:
            selected = [stop_frame]
        else:
            selected = np.linspace(stop_frame, last_frame, n_frames, dtype=int)

        selected = sorted(set(int(f) for f in selected))

        for fid in selected:
            if fid in raw_dict:
                all_frames.append(raw_dict[fid])

    return all_frames
