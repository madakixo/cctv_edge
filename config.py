"""Central configuration."""
from pathlib import Path
import os
import logging

BASE_DIR = Path(__file__).resolve().parent

# --- Paths ---
KNOWN_FACES_DIR = BASE_DIR / "known_faces"
OUTPUT_DIR = BASE_DIR / "outputs"
EVENT_FRAMES_DIR = OUTPUT_DIR / "event_frames"
DB_PATH = BASE_DIR / "cctv.db"

# --- Model ---
YOLO_MODEL = "yolov8n.pt"          # yolov8s.pt for better accuracy
DETECT_CONFIDENCE = 0.45
DETECT_CLASSES = [0]               # COCO: 0=person

# --- Face Recognition ---
FACE_TOLERANCE = 0.55              # Lower = stricter
FACE_DETECTION_MODEL = "hog"       # "cnn" for GPU accuracy (requires CUDA)
MIN_FACE_SIZE = 40                 # Skip tiny faces

# --- Stream / Video ---
STREAM_TIMEOUT = 10
FRAME_SKIP = 5                     # Process every 5th frame for performance with 29 cams
RESIZE_WIDTH = 640                 # Reduced for performance with 29 cams

# --- Source types ---
class SourceType:
    FOLDER   = "folder"
    FILE     = "file"
    STREAM   = "stream"
    DATABASE = "database"

for p in (OUTPUT_DIR, EVENT_FRAMES_DIR, KNOWN_FACES_DIR):
    p.mkdir(parents=True, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(BASE_DIR / "cctv.log")
    ]
)
