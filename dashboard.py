import cv2
import numpy as np


def _danger_color(level):
    r = int(255 * level)
    g = int(255 * (1.0 - level))
    return (0, g, r)


def draw_dashboard(annotated, pixel_speed, danger_lvl, last_accident_ts):
    h, w = annotated.shape[:2]

    y = 20
    for tid, speeds in list(pixel_speed.items())[:6]:
        if not speeds:
            continue
        spd = list(speeds.values())[-1]
        cv2.putText(annotated, f"Track {tid}: {spd:.1f} px/s", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        y += 18

    bar_h  = 22
    bar_w  = int(w * 0.45)
    bar_x  = 10
    bar_y  = h - bar_h - 10
    color  = _danger_color(danger_lvl)
    filled = int(bar_w * danger_lvl)

    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (40, 40, 40), -1)
    if filled > 0:
        cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + filled, bar_y + bar_h), color, -1)
    cv2.rectangle(annotated, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), (180, 180, 180), 1)
    cv2.putText(annotated, f"DANGER  {danger_lvl * 100:.0f}%", (bar_x + 4, bar_y + 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    label_x = bar_x + bar_w + 14
    label_y = bar_y + 15

    if danger_lvl >= 0.7:
        if last_accident_ts:
            cv2.putText(annotated, f"ACCIDENT CONFIRMED  {last_accident_ts}", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(annotated, "SUSPECTED - Verifying...", (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 165, 255), 1, cv2.LINE_AA)

    return annotated
