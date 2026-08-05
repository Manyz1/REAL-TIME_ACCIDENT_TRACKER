YOLO_MODEL    = 'yolov8s-seg.pt'
YOLO_CONF     = 0.40
YOLO_IOU      = 0.5
YOLO_CLASSES  = [2, 3, 5, 7]
YOLO_IMGSZ    = 640

WINDOW_SECONDS    = 10
MAX_FRAMES        = 300
ANALYZER_INTERVAL = 1.0

DROP_RATIO        = 0.2
STOP_RATIO        = 0.2
MIN_PEAK_SPEED    = 5
STOP_THRESHOLD    = 1.0
MEDFILT_KERNEL    = 25
CRASH_FRAME_COUNT = 5

CLIP_MODEL     = "openai/clip-vit-large-patch14"
CLIP_THRESHOLD = 0.85
SEVERE_PROMPTS = [
    "a photo of a severe car accident",
    "dashcam footage of a catastrophic traffic accident",
    "a photo of a deadly road accident",
    "a photo of a high speed collision",
    "a photo of a violent vehicle crash",
    "a vehicle completely crushed after a collision",
    "a photo of a totally wrecked vehicle",
    "a photo of a severe highway accident",
    "a serious multi-vehicle collision",
    "a car destroyed in a crash",
    "a photo of a rolled over vehicle",
    "an accident scene requiring emergency response",
]

NORMAL_PROMPTS = [
    "a photo of a normal driving car",
    "dashcam footage of a vehicle driving safely",
    "a photo of a parked vehicle",
    "normal highway traffic",
    "cars moving safely on the road",
    "a photo of a clean undamaged car",
    "vehicles waiting at a traffic light",
    "a photo of a normal highway scene",
]

CRASHES_DIR = "crashes"

CLOUD_YOLO_MODEL = "yolov8n.pt"
CLOUD_YOLO_IMGSZ = 416
