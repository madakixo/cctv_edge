"""Unified iterator for folder / file / stream / database sources."""
import os
import glob
import cv2
import time
import logging
from typing import Iterator, Tuple
from config import SourceType, STREAM_TIMEOUT, RESIZE_WIDTH
from database import get_enabled_sources

logger = logging.getLogger(__name__)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".m4v", ".wmv", ".flv"}

def _resize(frame):
    if frame is None:
        return None
    h, w = frame.shape[:2]
    if w > RESIZE_WIDTH:
        scale = RESIZE_WIDTH / w
        frame = cv2.resize(frame, (RESIZE_WIDTH, int(h * scale)))
    return frame

def _iter_video(cap, source_label, is_stream=False):
    reconnect_attempts = 0
    max_reconnects = 5

    while True:
        ok, frame = cap.read()
        if not ok:
            if is_stream and reconnect_attempts < max_reconnects:
                reconnect_attempts += 1
                logger.warning(f"[STREAM] {source_label} lost connection. Reconnecting {reconnect_attempts}/{max_reconnects}...")
                time.sleep(2)
                # cap.release() is usually handled by the caller or we can try to re-open here if we had the URI
                # But for simplicity in this iterator, we might need the URI.
                break # Let the upper layer handle re-opening for streams
            break
        reconnect_attempts = 0
        yield _resize(frame), source_label
    cap.release()

def iter_folder(folder_path: str) -> Iterator[Tuple]:
    files = []
    for ext in IMAGE_EXTS | VIDEO_EXTS:
        files += glob.glob(os.path.join(folder_path, f"**/*{ext}"), recursive=True)
    for fp in sorted(files):
        yield from _iter_media_file(fp, f"folder:{os.path.basename(fp)}")

def iter_file(file_path: str) -> Iterator[Tuple]:
    yield from _iter_media_file(file_path, f"file:{os.path.basename(file_path)}")

def _iter_media_file(file_path, label):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in IMAGE_EXTS:
        img = cv2.imread(file_path)
        if img is not None:
            yield _resize(img), label
    elif ext in VIDEO_EXTS:
        cap = cv2.VideoCapture(file_path)
        yield from _iter_video(cap, label)

def iter_stream(uri: str, name: str = "stream") -> Iterator[Tuple]:
    while True: # Keep stream alive
        cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if not cap.isOpened():
            logger.error(f"[STREAM] Failed to open: {uri}. Retrying in 5s...")
            time.sleep(5)
            continue

        logger.info(f"[STREAM] Connected: {uri}")
        yield from _iter_video(cap, f"{name}", is_stream=True)
        cap.release()
        logger.warning(f"[STREAM] Stream {name} ended. Restarting in 5s...")
        time.sleep(5)

def get_enabled_source_records():
    return get_enabled_sources()

def get_iterator(source_type: str, uri: str = None):
    """Factory that returns a fresh generator."""
    if source_type == SourceType.FOLDER:
        return iter_folder(uri)
    if source_type == SourceType.FILE:
        return iter_file(uri)
    if source_type == SourceType.STREAM:
        return iter_stream(uri)
    raise ValueError(f"Unknown source_type: {source_type}")
