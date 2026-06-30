"""Unified iterator for folder / file / stream / database sources."""
import os
import glob
import cv2
from typing import Iterator, Tuple
from config import SourceType, STREAM_TIMEOUT, RESIZE_WIDTH
from database import get_enabled_sources


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


def _iter_video(cap, source_label):
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        yield _resize(frame), source_label
    cap.release()


def iter_folder(folder_path: str) -> Iterator[Tuple, str]:
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
    cap = cv2.VideoCapture(uri, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    if not cap.isOpened():
        print(f"[STREAM] Failed to open: {uri}")
        return
    print(f"[STREAM] Connected: {uri}")
    yield from _iter_video(cap, f"stream:{name}")


def iter_database_sources() -> Iterator[Tuple]:
    """Each enabled row in `sources` table is dispatched by its source_type."""
    for src in get_enabled_sources():
        if src.source_type == SourceType.FOLDER:
            yield from iter_folder(src.uri)
        elif src.source_type == SourceType.FILE:
            yield from iter_file(src.uri)
        elif src.source_type == SourceType.STREAM:
            yield from iter_stream(src.uri, src.name)


def get_iterator(source_type: str, uri: str = None):
    """Factory that returns a fresh generator."""
    if source_type == SourceType.FOLDER:
        return iter_folder(uri)
    if source_type == SourceType.FILE:
        return iter_file(uri)
    if source_type == SourceType.STREAM:
        return iter_stream(uri)
    if source_type == SourceType.DATABASE:
        return iter_database_sources()
    raise ValueError(f"Unknown source_type: {source_type}")
