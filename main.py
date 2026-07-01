"""Entry point: CLI for selecting source type."""
import argparse
import sys
from database import init_db, Session, SourceRecord
from pipeline import CctvPipeline
from config import SourceType


def seed_sample_sources():
    """Optional: insert sample source rows into the DB."""
    with Session() as s:
        if s.query(SourceRecord).count() > 0:
            return
        s.add_all([
            SourceRecord(name="Front Door Cam",
                         source_type=SourceType.STREAM,
                         uri="rtsp://admin:pass@192.168.1.10:554/Streaming/Channels/101",
                         description="RTSP IP camera"),
            SourceRecord(name="CCTV Footage",
                         source_type=SourceType.FILE,
                         uri="./samples/cctv_clip.mp4"),
            SourceRecord(name="Photo Folder",
                         source_type=SourceType.FOLDER,
                         uri="./samples/images"),
        ])
        s.commit()
        print("[DB] Seeded sample sources.")


def main():
    parser = argparse.ArgumentParser(description="CCTV Object + Face Recognition")
    parser.add_argument("--source", choices=["folder", "file", "stream", "database"],
                        default="database")
    parser.add_argument("--uri", help="Path/URL (ignored for database source)")
    parser.add_argument("--no-show", action="store_true",
                        help="Disable GUI window")
    args = parser.parse_args()

    init_db()
    seed_sample_sources()

    pipeline = CctvPipeline()
    try:
        pipeline.run(args.source, args.uri, show=not args.no_show)
    except KeyboardInterrupt:
        print("\n[EXIT] Interrupted by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
