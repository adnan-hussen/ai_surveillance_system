"""Central configuration for the Zero-Shot Video Analytics Surveillance Pipeline."""

from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path

BASE_DIR = Path(__file__).parent.resolve()
DATA_DIR = BASE_DIR / "data"
LANCEDB_URI = str(DATA_DIR / "lancedb_store")
INCIDENT_DB_PATH = str(DATA_DIR / "verified_incidents.db")


@dataclass
class PipelineConfig:
    # --- 1. Person Detector & Multi-Object Tracker (Ultralytics YOLO26s + ByteTrack) ---
    yolo_model_path: str = os.getenv("YOLO_MODEL", "yolo26m.pt")
    person_conf_threshold: float = 0.7
    person_iou_threshold: float = 0.50
    tracker_config: str = "bytetrack.yaml"
    crop_padding_ratio: float = float(os.getenv("CROP_PADDING_RATIO", "0.35"))  # 35% context padding around bbox
    min_crop_size: int = 64

    # --- 2. System-One Decision Model (LiquidAI/d1-3B) ---
    decision_model_id: str = os.getenv("DECISION_MODEL_ID", "LiquidAI/d1-3B")
    decision_input_side: int = 512
    compile_decision_model: bool = False
    default_detection_rule: str = "Is the person in the frame smoking?"
    decision_threshold: float = float(os.getenv("DECISION_THRESHOLD", "0.65"))  # Positive match probability threshold
    decision_eval_interval_sec: float = 0.50  # Minimum interval between VLM evaluations per track_id

    # --- 3. Person Re-ID Model (OSNet: Omni-Scale Feature Extractor) ---
    reid_model_name: str = os.getenv("REID_MODEL_NAME", "osnet_x1_0_msmt17")
    reid_weights_path: str = os.getenv("REID_WEIGHTS", str(DATA_DIR / "models" / "osnet_x1_0_msmt17.pt"))
    reid_input_size: tuple[int, int] = (256, 128)
    reid_match_cosine_threshold: float = float(os.getenv("REID_MATCH_THRESHOLD", "0.65"))
    embedding_dim: int = 512

    # Backward compatibility aliases for face recognition legacy fields
    face_model_pack: str = os.getenv("FACE_MODEL_PACK", "buffalo_s")
    face_det_size: tuple[int, int] = (640, 640)
    face_det_threshold: float = 0.55
    face_match_cosine_threshold: float = reid_match_cosine_threshold

    # --- 4. Vector Database (LanceDB) & Watchlist Retention Policy ---
    lancedb_uri: str = LANCEDB_URI
    lancedb_table_name: str = "subject_embeddings"
    face_match_cosine_threshold: float = 0.55  # Cosine similarity threshold for cross-camera Re-ID
    verification_timeout_sec: float = float(os.getenv("VERIFICATION_TIMEOUT_SEC", "45.0"))
    ttl_cleanup_interval_sec: float = 2.0

    # --- 5. Event Service & Incident Frame Buffering ---
    incident_db_path: str = INCIDENT_DB_PATH
    max_frames_per_subject: int = 10  # 10 frames captured during the detection alert window
    frame_capture_interval_sec: float = 0.12  # 120ms burst interval (~1.2s incident sequence)
    crop_jpeg_quality: int = 85

    # --- 6. Camera Sources ---
    # Map camera_id -> OpenCV source (RTSP URL, device index like 0, or video file path)
    camera_sources: dict[str, str | int] = field(default_factory=lambda: {
        "CAM-01 (Primary Feed)": int(os.getenv("CAMERA_INDEX", "0"))
    })
    target_fps: int = 30
