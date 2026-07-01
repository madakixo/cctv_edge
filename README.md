###  cctv_edge
Modular Python system. It reads from folder, file path, stream (RTSP/HTTP/webcam), and database, runs YOLOv8 object detection + face recognition/identification, and logs all events back to a database.
# CCTV Object Detection & Facial Recognition System

A modular, production-ready Python pipeline for real-time and batch CCTV analysis. It supports multiple input sources (Folders, Files, Live Streams, and Databases), detects objects using YOLOv8, and performs facial recognition/identification with `face_recognition`. All events and analytics are logged to a database.

---

## Key Features

- **Multi-Source Input**:
  - **Folder**: Recursively processes all images and videos in a directory.
  - **File**: Processes a single video or image file.
  - **Stream**: Connects to live RTSP, HTTP, or local webcam streams.
  - **Database**: Dynamically reads enabled camera/source configurations from a database table.
- **Object Detection**: State-of-the-art YOLOv8 for detecting people, vehicles, and more.
- **Facial Recognition**: Identifies known individuals and flags unknown faces using dlib.
- **Auto-Enrollment**: Drop an image named `person_name.jpg` into the `known_faces/` folder, and the system automatically encodes and registers them in the database on the next run.
- **Event Logging**: Every detection and recognition event is logged to an SQLite database (easily upgradable to PostgreSQL/MySQL) with timestamps and bounding boxes.
- **Audit Snapshots**: Saves frame snapshots to disk whenever a face is recognized for later review.
- **Headless Mode**: Run on servers without a display using the `--no-show` flag.

---

## Project Structure

```text
cctv_system/
├── config.py             # Central configuration (models, paths, thresholds)
├── database.py           # SQLAlchemy ORM and DB operations
├── sources.py            # Unified iterators for folder/file/stream/db
├── detector.py           # YOLOv8 wrapper
├── face_engine.py        # Face recognition and auto-enrollment logic
├── pipeline.py           # Main processing loop combining detection + faces
├── main.py               # CLI entry point
├── known_faces/          # Place known face images here (e.g., john_doe.jpg)
├── outputs/              # Generated audit frames are saved here
├── cctv.db               # SQLite database (auto-created)
└── requirements.txt
```

---

## 🛠 Installation

### 1. Clone the Repository
```bash
git clone https://your-repo-link.git
cd cctv_system
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

**Standard Python packages:**
```bash
pip install -r requirements.txt
```

**Important Note on `face_recognition` (dlib):**
The `face_recognition` library requires `dlib`. If the pip install fails, you need to install CMake and dlib manually:
- **Ubuntu/Debian:** `sudo apt-get install cmake libdlib-dev` then `pip install face-recognition`
- **macOS:** `brew install cmake dlib` then `pip install face-recognition`
- **Windows:** Install Visual Studio Build Tools, install CMake, and then `pip install dlib face-recognition`.

---

## Configuration

All settings are centralized in `config.py`. 

| Setting | Description | Default |
|---|---|---|
| `YOLO_MODEL` | YOLOv8 weight file. | `yolov8n.pt` |
| `DETECT_CONFIDENCE` | Minimum confidence threshold for objects. | `0.45` |
| `DETECT_CLASSES` | COCO classes to detect (0=person). | `[0]` |
| `FACE_TOLERANCE` | Strictness of face match (lower is stricter). | `0.55` |
| `FACE_DETECTION_MODEL` | `hog` (CPU/fast) or `cnn` (GPU/accurate). | `hog` |
| `FRAME_SKIP` | Processes every Nth frame to save CPU. | `3` |
| `RESIZE_WIDTH` | Scales frames to this width before processing. | `1280` |

---

## Usage

### 1. Register Known Faces
Place clear, front-facing images of individuals into the `known_faces/` directory. Name the files using the person's name:
```text
known_faces/
├── john_doe.jpg
├── jane_smith.png
└── admin.jpg
```

### 2. Run the Pipeline

Use the `main.py` CLI to start processing. 

```bash
# 1. Process sources defined in the database
python main.py --source database

# 2. Process a single video file (e.g., ./data/cctv_clip.mp4)
python main.py --source file --uri ./data/cctv_clip.mp4

# 3. Process a folder of images/videos
python main.py --source folder --uri ./data/images/

# 4. Connect to a live RTSP stream
python main.py --source stream --uri "rtsp://user:pass@192.168.1.10:554/stream"

# 5. Use local webcam
python main.py --source stream --uri "0"

# 6. Run in headless mode (for servers / Docker)
python main.py --source database --no-show
```

*(Note: The script automatically seeds the database with sample source records on the first run. You can edit these directly in the `sources` table or via your own scripts.)*

---

## 🗄 Database Schema

The system uses SQLAlchemy and creates an SQLite file (`cctv.db`) automatically. 

### `known_faces`
Stores facial encodings.
- `name`: Name of the person.
- `encoding`: Binary blob of the 128D face vector.
- `image_path`: Path to the original enrollment image.

### `sources`
Configurable list of inputs for the `database` source type.
- `name`: Friendly name of the camera/file.
- `source_type`: `folder`, `file`, or `stream`.
- `uri`: File path or RTSP URL.
- `enabled`: `1` (Active) or `0` (Disabled).

### `cctv_events`
Audit log of all detections.
- `timestamp`: When the event occurred.
- `source` / `source_type`: Where the frame came from.
- `object_class`: What was detected (e.g., `person`, `face`).
- `face_name`: Identified person name (or `Unknown`).
- `confidence` / `face_distance`: Algorithm confidence scores.
- `frame_path`: Path to the saved JPEG snapshot (for faces).
- `bbox`: Bounding box coordinates (`x1,y1,x2,y2`).

### Querying Events Example
```sql
SELECT timestamp, source, face_name, frame_path 
FROM cctv_events 
WHERE face_name = 'john_doe' 
ORDER BY timestamp DESC;
```

---

## Scaling to Production

To scale this system for enterprise or large-scale deployments:
1. **Database**: Swap SQLite for PostgreSQL by updating the `create_engine` string in `database.py`.
2. **GPU Acceleration**: Change `FACE_DETECTION_MODEL` to `"cnn"` in `config.py` and run on a CUDA-enabled machine.
3. **Message Queues**: Replace the synchronous `pipeline.py` loop with a task queue (like Celery or RabbitMQ) to distribute frame processing across multiple worker nodes.
4. **Streaming Output**: Instead of `cv2.imshow()`, write annotated frames back to an RTSP server using FFmpeg or GStreamer.
