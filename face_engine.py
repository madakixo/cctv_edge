"""Face recognition / identification engine."""
import os
import logging
import numpy as np
import face_recognition
import cv2
import threading
from config import (KNOWN_FACES_DIR, FACE_TOLERANCE, FACE_DETECTION_MODEL,
                    MIN_FACE_SIZE)
from database import load_known_faces, add_known_face

logger = logging.getLogger(__name__)

class FaceEngine:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(FaceEngine, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self.known_encodings = []
        self.known_names = []
        self.tolerance = FACE_TOLERANCE
        self.model = FACE_DETECTION_MODEL
        self.reload_lock = threading.Lock()
        self._load_known()
        self._initialized = True

    def _load_known(self):
        with self.reload_lock:
            self.known_encodings = []
            self.known_names = []

            # 1) From DB
            try:
                db_faces = load_known_faces()
                for name, enc in db_faces:
                    if name not in self.known_names:
                        self.known_encodings.append(enc)
                        self.known_names.append(name)
            except Exception as e:
                logger.error(f"Error loading faces from DB: {e}")

            # 2) Auto-register any new images dropped into known_faces/
            if os.path.isdir(KNOWN_FACES_DIR):
                for fn in os.listdir(KNOWN_FACES_DIR):
                    if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                        continue
                    name = os.path.splitext(fn)[0]
                    if name in self.known_names:
                        continue
                    path = os.path.join(KNOWN_FACES_DIR, fn)
                    try:
                        img = face_recognition.load_image_file(path)
                        encs = face_recognition.face_encodings(img)
                        if encs:
                            enc = encs[0]
                            self.known_encodings.append(enc)
                            self.known_names.append(name)
                            add_known_face(name, enc, path)
                            logger.info(f"[FACE] Registered new identity: {name}")
                    except Exception as e:
                        logger.error(f"Failed to process face image {path}: {e}")

            logger.info(f"[FACE] Loaded {len(self.known_names)} known identities.")

    def reload(self):
        logger.info("Reloading known faces...")
        self._load_known()

    def recognize(self, frame_bgr):
        """Returns list of dicts: {name, distance, bbox:[top,right,bottom,left]}."""
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            locations = face_recognition.face_locations(rgb, model=self.model)
            encodings = face_recognition.face_encodings(rgb, locations)
        except Exception as e:
            logger.error(f"Face detection/recognition error: {e}")
            return []

        results = []
        # Use local references to avoid issues if reload happens mid-loop
        with self.reload_lock:
            current_encodings = list(self.known_encodings)
            current_names = list(self.known_names)

        for loc, enc in zip(locations, encodings):
            top, right, bottom, left = loc
            if (right - left) < MIN_FACE_SIZE or (bottom - top) < MIN_FACE_SIZE:
                continue

            name, distance = "Unknown", None
            if current_encodings:
                distances = face_recognition.face_distance(current_encodings, enc)
                best_idx = int(np.argmin(distances))
                if distances[best_idx] <= self.tolerance:
                    name = current_names[best_idx]
                    distance = float(distances[best_idx])

            results.append({
                "name": name,
                "distance": distance,
                "bbox": [top, right, bottom, left],
            })
        return results
