"""YOLOv8 object detection wrapper."""
import numpy as np
import cv2
import logging
from ultralytics import YOLO
from config import YOLO_MODEL, DETECT_CONFIDENCE, DETECT_CLASSES

logger = logging.getLogger(__name__)

class ObjectDetector:
    def __init__(self, model_path: str = YOLO_MODEL):
        logger.info(f"[YOLO] Loading model {model_path} ...")
        try:
            self.model = YOLO(model_path)
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            raise
        self.classes = DETECT_CLASSES
        self.conf = DETECT_CONFIDENCE

    def detect(self, frame: np.ndarray):
        """Return list of dicts: {class_name, confidence, bbox:[x1,y1,x2,y2]}."""
        try:
            results = self.model.predict(
                frame, conf=self.conf, classes=self.classes, verbose=False
            )
        except Exception as e:
            logger.error(f"YOLO prediction error: {e}")
            return []

        detections = []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                cls_id = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                detections.append({
                    "class_id": cls_id,
                    "class_name": self.model.names[cls_id],
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                })
        return detections

    @staticmethod
    def crop(frame, bbox):
        x1, y1, x2, y2 = bbox
        return frame[y1:y2, x1:x2]
