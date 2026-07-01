"""Face recognition / identification engine."""
import os
import pickle
import numpy as np
import face_recognition
from config import (KNOWN_FACES_DIR, FACE_TOLERANCE, FACE_DETECTION_MODEL,
                    MIN_FACE_SIZE)
from database import load_known_faces, add_known_face


class FaceEngine:
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.tolerance = FACE_TOLERANCE
        self.model = FACE_DETECTION_MODEL
        self._load_known()

    def _load_known(self):
        # 1) From DB
        db_faces = load_known_faces()
        for name, enc in db_faces:
            self.known_encodings.append(enc)
            self.known_names.append(name)

        # 2) Auto-register any new images dropped into known_faces/
        if os.path.isdir(KNOWN_FACES_DIR):
            for fn in os.listdir(KNOWN_FACES_DIR):
                if not fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                name = os.path.splitext(fn)[0]
                if name in self.known_names:
                    continue
                path = os.path.join(KNOWN_FACES_DIR, fn)
                img = face_recognition.load_image_file(path)
                encs = face_recognition.face_encodings(img)
                if encs:
                    enc = encs[0]
                    self.known_encodings.append(enc)
                    self.known_names.append(name)
                    add_known_face(name, enc, path)
                    print(f"[FACE] Registered new identity: {name}")

        print(f"[FACE] Loaded {len(self.known_names)} known identities.")

    def recognize(self, frame_bgr):
        """Returns list of dicts: {name, distance, bbox:[top,right,bottom,left]}."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB) \
            if frame_bgr.dtype == np.uint8 else frame_bgr
        h, w = rgb.shape[:2]
        locations = face_recognition.face_locations(rgb, model=self.model)
        encodings = face_recognition.face_encodings(rgb, locations)

        results = []
        for loc, enc in zip(locations, encodings):
            top, right, bottom, left = loc
            if (right - left) < MIN_FACE_SIZE or (bottom - top) < MIN_FACE_SIZE:
                continue
            name, distance = "Unknown", None
            if self.known_encodings:
                distances = face_recognition.face_distance(
                    self.known_encodings, enc)
                best_idx = int(np.argmin(distances))
                if distances[best_idx] <= self.tolerance:
                    name = self.known_names[best_idx]
                    distance = float(distances[best_idx])
            results.append({
                "name": name,
                "distance": distance,
                "bbox": [top, right, bottom, left],  # face_recognition format
            })
        return results


import cv2  # placed here to avoid top-of-file clutter
