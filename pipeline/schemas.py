"""Data structures and schemas for the video analytics surveillance pipeline."""

from __future__ import annotations
import time
from dataclasses import dataclass, field, asdict
from typing import Literal

SubjectStatus = Literal["unverified", "verified", "dismissed", "expired"]


@dataclass
class BoundingBox:
    x1: int
    y1: int
    x2: int
    y2: int
    padded_x1: int = 0
    padded_y1: int = 0
    padded_x2: int = 0
    padded_y2: int = 0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrackedSubject:
    camera_id: str
    track_id: int
    bbox: BoundingBox
    det_confidence: float
    frame_width: int = 1280
    frame_height: int = 720
    subject_id: str | None = None
    subject_status: SubjectStatus | None = None
    match_probability: float | None = None
    detection_rule: str | None = None
    face_detected: bool = False
    reid_similarity: float | None = None

    def to_dict(self) -> dict:
        return {
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "bbox": self.bbox.to_dict(),
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "det_confidence": round(self.det_confidence, 4),
            "subject_id": self.subject_id,
            "subject_status": self.subject_status,
            "match_probability": round(self.match_probability, 4) if self.match_probability is not None else None,
            "detection_rule": self.detection_rule,
            "face_detected": self.face_detected,
            "reid_similarity": round(self.reid_similarity, 4) if self.reid_similarity is not None else None,
        }


@dataclass
class SubjectFrameRecord:
    """Single incident frame captured for a flagged subject during the detection window."""
    frame_id: str
    subject_id: str
    camera_id: str
    track_id: int
    timestamp: float
    match_probability: float
    crop_b64: str        # Base64 JPEG of the context-padded subject crop (with thin bbox overlay)
    full_frame_b64: str  # Base64 JPEG thumbnail of the full camera scene (with thin bbox overlay)
    bbox: dict | None = None
    face_crop_b64: str | None = None

    def to_dict(self) -> dict:
        return {
            "frame_id": self.frame_id,
            "subject_id": self.subject_id,
            "camera_id": self.camera_id,
            "track_id": self.track_id,
            "timestamp": self.timestamp,
            "match_probability": round(self.match_probability, 4),
            "crop_b64": self.crop_b64,
            "full_frame_b64": self.full_frame_b64,
            "bbox": self.bbox,
            "face_crop_b64": self.face_crop_b64,
        }


@dataclass
class FlaggedSubjectRecord:
    """Active state of an unverified or verified flagged subject tracked across cameras."""
    subject_id: str
    status: SubjectStatus
    detection_rule: str
    initial_confidence: float
    peak_confidence: float
    created_at: float
    expires_at: float | None  # None once verified; unix timestamp while status == "unverified"
    verified_at: float | None = None
    cameras_seen: list[str] = field(default_factory=list)
    active_tracks: dict[str, int] = field(default_factory=dict)  # camera_id -> track_id
    latest_bbox: dict | None = None
    frame_width: int = 1280
    frame_height: int = 720
    has_face_embedding: bool = False
    face_crop_b64: str | None = None
    frames: list[SubjectFrameRecord] = field(default_factory=list)

    def remaining_ttl(self, now: float | None = None) -> float | None:
        if self.status != "unverified" or self.expires_at is None:
            return None
        now = now or time.time()
        return max(0.0, round(self.expires_at - now, 1))

    def to_dict(self, include_frames: bool = True) -> dict:
        now = time.time()
        data = {
            "subject_id": self.subject_id,
            "status": self.status,
            "detection_rule": self.detection_rule,
            "initial_confidence": round(self.initial_confidence, 4),
            "peak_confidence": round(self.peak_confidence, 4),
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "remaining_ttl_sec": self.remaining_ttl(now),
            "verified_at": self.verified_at,
            "cameras_seen": list(self.cameras_seen),
            "active_tracks": dict(self.active_tracks),
            "latest_bbox": self.latest_bbox,
            "frame_width": self.frame_width,
            "frame_height": self.frame_height,
            "has_face_embedding": self.has_face_embedding,
            "face_crop_b64": self.face_crop_b64,
            "frame_count": len(self.frames),
        }
        if include_frames:
            data["frames"] = [f.to_dict() for f in self.frames]
        return data
