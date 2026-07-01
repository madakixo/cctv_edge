import argparse
import sys
import logging
import concurrent.futures
from database import init_db, Session, SourceRecord
from pipeline import CctvPipeline
from config import SourceType
from sources import get_enabled_source_records

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def seed_sample_sources():
    """Optional: insert sample source rows into the DB."""
    with Session() as s:
        if s.query(SourceRecord).count() > 0:
            return
        s.add_all([
            SourceRecord(name="Sample Cam 1",
                         source_type=SourceType.STREAM,
                         uri="0", # Local webcam
                         description="Local Webcam"),
        ])
        s.commit()
        logger.info("[DB] Seeded sample sources.")

def run_pipeline_for_source(source_record, no_show):
    pipeline = CctvPipeline()
    try:
        pipeline.run_source(
            source_type=source_record.source_type,
            uri=source_record.uri,
            label=source_record.name,
            show=not no_show
        )
    except Exception as e:
        logger.error(f"Error in pipeline for {source_record.name}: {e}")

def main():
    parser = argparse.ArgumentParser(description="Production CCTV Object + Face Recognition")
    parser.add_argument("--source", choices=["folder", "file", "stream", "database"],
                        default="database")
    parser.add_argument("--uri", help="Path/URL (ignored for database source)")
    parser.add_argument("--no-show", action="store_true", default=True,
                        help="Disable GUI window (default: True for production)")
    parser.add_argument("--max-workers", type=int, default=30,
                        help="Max parallel camera streams")
    args = parser.parse_args()

    init_db()

    if args.source == "database":
        seed_sample_sources()
        sources = get_enabled_source_records()
    else:
        # Create a transient SourceRecord for CLI-passed URI
        sources = [SourceRecord(name=args.source, source_type=args.source, uri=args.uri)]

    logger.info(f"Starting pipeline with {len(sources)} sources...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        futures = [executor.submit(run_pipeline_for_source, src, args.no_show) for src in sources]
        try:
            concurrent.futures.wait(futures)
        except KeyboardInterrupt:
            logger.info("Interrupted by user, shutting down...")
            executor.shutdown(wait=False)
            sys.exit(0)

if __name__ == "__main__":
    main()
