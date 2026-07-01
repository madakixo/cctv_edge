"""Main CCTV processing pipeline combining detection + face recognition."""
import os
import uuid
import cv2
import logging
import threading
from config import (FRAME_SKIP, EVENT_FRAMES_DIR)
from detector import ObjectDetector
from face_engine import FaceEngine
from database import log_event

logger = logging.getLogger(__name__)

class CctvPipeline:
    def __init__(self):
        self.detector = ObjectDetector()
        self.face_engine = FaceEngine() # Singleton engine
        self.frame_counter = 0

    @staticmethod
    def _save_frame(frame, prefix="event"):
        fname = f"{prefix}_{uuid.uuid4().hex[:8]}.jpg"
        path = os.path.join(EVENT_FRAMES_DIR, fname)
        # Using cv2.imwrite is generally thread-safe as it's a stateless I/O operation
        cv2.imwrite(path, frame)
        return path

    def _draw(self, frame, detections, faces):
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{d['class_name']} {d['confidence']:.2f}",
                        (x1, max(15, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        for f in faces:
            top, right, bottom, left = f["bbox"]
            color = (0, 0, 255) if f["name"] == "Unknown" else (255, 0, 0)
            cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
            label = f["name"]
            if f["distance"] is not None:
                label += f" ({1 - f['distance']:.2f})"
            cv2.putText(frame, label, (left, max(15, top - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        return frame

    def process_frame(self, frame, source_label, source_type):
        if frame is None:
            return None
        self.frame_counter += 1
        # In multi-camera setup, frame skip helps reducing overall CPU load
        if self.frame_counter % FRAME_SKIP != 0:
            return None

        # Process frame
        detections = self.detector.detect(frame)
        faces = self.face_engine.recognize(frame)

        # Log every detection
        for d in detections:
            log_event(
                source=source_label,
                source_type=source_type,
                object_class=d["class_name"],
                confidence=d["confidence"],
                bbox=",".join(map(str, d["bbox"])),
            )

        # Log each face match (known or unknown)
        for f in faces:
            frame_path = self._save_frame(frame, prefix=f"face_{f['name']}")
            log_event(
                source=source_label,
                source_type=source_type,
                object_class="face",
                confidence=1.0 - (f["distance"] or 0.0),
                face_name=f["name"],
                face_distance=f["distance"],
                frame_path=frame_path,
                bbox=",".join(map(str, f["bbox"])),
            )

        return (detections, faces)

    def run_source(self, source_type: str, uri: str, label: str, show: bool = False):
        from sources import get_iterator
        logger.info(f"Starting pipeline for {label}")
        try:
            iterator = get_iterator(source_type, uri)
            for frame, _ in iterator:
                res = self.process_frame(frame, label, source_type)
                if res and show:
                    detections, faces = res
                    out = self._draw(frame, detections, faces)
                    cv2.imshow(f"CCTV: {label}", out)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
        except Exception as e:
            logger.error(f"Error in pipeline {label}: {e}")
        finally:
            if show:
                try:
                    cv2.destroyWindow(f"CCTV: {label}")
                except:
                    pass
            logger.info(f"Finished pipeline for {label}")
