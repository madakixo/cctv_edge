"""Database layer using SQLAlchemy (SQLite by default, easy to swap)."""
import datetime
import numpy as np
import logging
from sqlalchemy import (
    create_engine, Column, Integer, String, Float, DateTime, LargeBinary, Text, event
)
from sqlalchemy.orm import declarative_base, sessionmaker
from config import DB_PATH

logger = logging.getLogger(__name__)

Base = declarative_base()
# Increase timeout for SQLite to handle multiple concurrent writers better
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,
    connect_args={'timeout': 30},
    pool_size=20,
    max_overflow=10
)
Session = sessionmaker(bind=engine)

@event.listens_for(engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()

class KnownFace(Base):
    __tablename__ = "known_faces"
    id          = Column(Integer, primary_key=True)
    name        = Column(String, nullable=False, index=True)
    encoding    = Column(LargeBinary, nullable=False)   # pickled ndarray
    image_path  = Column(Text)
    created_at  = Column(DateTime, default=datetime.datetime.utcnow)

class CCTVEvent(Base):
    __tablename__ = "cctv_events"
    id           = Column(Integer, primary_key=True)
    timestamp    = Column(DateTime, default=datetime.datetime.utcnow, index=True)
    source       = Column(String, index=True)
    source_type  = Column(String, index=True)
    object_class = Column(String, index=True)
    confidence   = Column(Float)
    face_name    = Column(String, nullable=True, index=True)
    face_distance = Column(Float, nullable=True)
    frame_path   = Column(Text, nullable=True)
    bbox         = Column(Text)              # "x1,y1,x2,y2"

class SourceRecord(Base):
    """Sources stored in DB (cameras, file paths, etc.)."""
    __tablename__ = "sources"
    id          = Column(Integer, primary_key=True)
    name        = Column(String, index=True)
    source_type = Column(String)
    uri         = Column(Text)               # path/rtsp url/file
    enabled     = Column(Integer, default=1)
    description = Column(Text)

def init_db():
    Base.metadata.create_all(engine)

def add_known_face(name: str, encoding: np.ndarray, image_path: str = None):
    with Session() as s:
        face = KnownFace(
            name=name,
            encoding=encoding.tobytes(),
            image_path=image_path,
        )
        s.add(face)
        s.commit()

def load_known_faces():
    """Returns list of (name, encoding_ndarray)."""
    with Session() as s:
        rows = s.query(KnownFace).all()
        return [(r.name, np.frombuffer(r.encoding, dtype=np.float64))
                for r in rows]

def log_event(**kwargs):
    try:
        with Session() as s:
            event = CCTVEvent(**kwargs)
            s.add(event)
            s.commit()
            return event.id
    except Exception as e:
        logger.error(f"Failed to log event: {e}")
        return None

def get_enabled_sources():
    with Session() as s:
        return s.query(SourceRecord).filter(SourceRecord.enabled == 1).all()
