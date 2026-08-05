import os
import queue
import threading
import datetime
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from config import CLIP_MODEL, CLIP_THRESHOLD, SEVERE_PROMPTS, NORMAL_PROMPTS, CRASHES_DIR

_PROMPTS  = SEVERE_PROMPTS + NORMAL_PROMPTS
_N_SEVERE = len(SEVERE_PROMPTS)

crash_queue      = queue.Queue()
danger_level     = [0.0]
last_accident_ts = [None]

_device = "cuda" if torch.cuda.is_available() else "cpu"
_clip_model     = None
_clip_processor = None


def load_clip():
    global _clip_model, _clip_processor
    model_name = CLIP_MODEL if _device == "cuda" else "openai/clip-vit-base-patch32"
    _clip_model     = CLIPModel.from_pretrained(model_name).to(_device)
    _clip_processor = CLIPProcessor.from_pretrained(model_name)


def _verify(frames):
    if not frames or _clip_model is None:
        return 0.0, None
    best_score = 0.0
    best_frame = frames[0]
    for frame in frames:
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        inputs = _clip_processor(text=_PROMPTS, images=img, return_tensors="pt", padding=True).to(_device)
        with torch.no_grad():
            probs = _clip_model(**inputs).logits_per_image.softmax(dim=-1).cpu().numpy()[0]
        severe_score = float(probs[:_N_SEVERE].sum())
        if severe_score > best_score:
            best_score = severe_score
            best_frame = frame
    return best_score, best_frame


def _worker():
    os.makedirs(CRASHES_DIR, exist_ok=True)
    while True:
        item = crash_queue.get()
        if item is None:
            break
        try:
            frames    = item[0]
            ts        = item[1]
            threshold = item[2] if len(item) > 2 else CLIP_THRESHOLD
            score, best_frame = _verify(frames)
            if score >= threshold:
                danger_level[0]     = 1.0
                last_accident_ts[0] = ts
                if best_frame is not None:
                    for old in os.listdir(CRASHES_DIR):
                        os.remove(os.path.join(CRASHES_DIR, old))
                    cv2.imwrite(f"{CRASHES_DIR}/{ts}.png", best_frame)
                print(f"[ACCIDENT CONFIRMED] {ts}  score={score:.2f}")
        finally:
            crash_queue.task_done()


def start_clip_thread():
    t = threading.Thread(target=_worker, daemon=True)
    t.start()
