"""Event Service: Subject Watchlist, Cross-Camera Re-ID, Incident Frame Buffering & Incident Archive DB."""

from __future__ import annotations
import base64
import io
import json
import sqlite3
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Callable
import cv2
import numpy as np
from PIL import Image, ImageDraw

from config import PipelineConfig
from pipeline.schemas import BoundingBox, FlaggedSubjectRecord, SubjectFrameRecord
from pipeline.vector_db import SubjectVectorDB


def pil_to_b64_jpeg(img: Image.Image, quality: int = 85, max_side: int | None = None) -> str:
    """Encodes a PIL Image to a base64 JPEG data URL."""
    work = img.convert("RGB")
    if max_side is not None:
        work.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    work.save(buf, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def bgr_to_b64_jpeg(frame_bgr: np.ndarray, quality: int = 80, max_side: int | None = None) -> str:
    """Encodes an OpenCV BGR array to a base64 JPEG data URL."""
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    return pil_to_b64_jpeg(Image.fromarray(rgb), quality=quality, max_side=max_side)


def render_annotated_scene_b64(
    full_frame_bgr: np.ndarray,
    bbox: BoundingBox,
    subject_id: str,
    status: str,
    match_probability: float,
    quality: int = 78,
    max_side: int = 540,
) -> str:
    """Draws a crisp, thin bounding box on a copy of the full scene frame for incident logs."""
    annotated = full_frame_bgr.copy()
    h, w = annotated.shape[:2]
    # Scale stroke so after resizing to max_side=540 it renders as a clean thin 1.5px-2px box
    thin_stroke = max(1, int(round(max(w, h) / 480.0)))
    # BGR: Red (68, 68, 239) for verified, Yellow (8, 179, 234) for unverified
    color_bgr = (68, 68, 239) if status == "verified" else (8, 179, 234)

    cv2.rectangle(
        annotated,
        (bbox.x1, bbox.y1),
        (bbox.x2, bbox.y2),
        color_bgr,
        thin_stroke,
    )

    label = f"{subject_id} ({match_probability * 100:.0f}%)"
    font_scale = max(0.42, max(w, h) / 1800.0)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
    ly1 = max(0, bbox.y1 - th - 8)
    cv2.rectangle(annotated, (bbox.x1, ly1), (bbox.x1 + tw + 8, ly1 + th + 6), color_bgr, -1)
    text_color = (255, 255, 255) if status == "verified" else (10, 11, 14)
    cv2.putText(
        annotated,
        label,
        (bbox.x1 + 4, ly1 + th + 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        text_color,
        1,
        cv2.LINE_AA,
    )
    return bgr_to_b64_jpeg(annotated, quality=quality, max_side=max_side)


def render_annotated_crop_b64(
    padded_crop_pil: Image.Image,
    bbox: BoundingBox,
    status: str,
    quality: int = 85,
    max_side: int = 280,
) -> str:
    """Draws a thin bounding box around the detected subject inside the context-padded crop."""
    annotated_pil = padded_crop_pil.copy().convert("RGB")
    draw = ImageDraw.Draw(annotated_pil)

    # Calculate tight subject box coordinates relative to the padded crop origin
    ix1 = max(0, bbox.x1 - bbox.padded_x1)
    iy1 = max(0, bbox.y1 - bbox.padded_y1)
    ix2 = min(annotated_pil.width - 1, bbox.x2 - bbox.padded_x1)
    iy2 = min(annotated_pil.height - 1, bbox.y2 - bbox.padded_y1)

    stroke_hex = "#ef4444" if status == "verified" else "#eab308"
    if ix2 > ix1 + 4 and iy2 > iy1 + 4:
        draw.rectangle([ix1, iy1, ix2, iy2], outline=stroke_hex, width=2)

    return pil_to_b64_jpeg(annotated_pil, quality=quality, max_side=max_side)


class EventService:
    """Central Security Event Service coordinating:
      - Active Detection Rule & Confidence Threshold state
      - Cross-camera track-to-subject mapping (`(camera_id, track_id) -> subject_id`)
      - LanceDB biometric vector queries and retention lifecycle (unverified -> verified / expired)
      - 10-frame incident window capture (with thin bounding boxes drawn on every logged frame)
      - Persistent SQLite Incident Archive Database for verified alerts
    """

    def __init__(
        self,
        config: PipelineConfig,
        vector_db: SubjectVectorDB,
        on_event_callback: Callable[[dict[str, Any]], None] | None = None,
    ):
        self.config = config
        self.vector_db = vector_db
        self.on_event_callback = on_event_callback
        self._lock = threading.RLock()

        self.current_detection_rule: str = config.default_detection_rule
        self.decision_threshold: float = config.decision_threshold
        self.verification_timeout_sec: float = config.verification_timeout_sec

        # Active unverified and verified flagged subjects: subject_id -> FlaggedSubjectRecord
        self.flagged_subjects: dict[str, FlaggedSubjectRecord] = {}
        # Fast lookup: (camera_id, track_id) -> subject_id
        self.track_to_subject: dict[tuple[str, int], str] = {}
        # Last frame capture timestamp per subject_id to space out the 10-frame burst
        self._last_frame_saved_at: dict[str, float] = {}

        self._init_incident_db()

    def _init_incident_db(self) -> None:
        """Initializes the SQLite Incident Database for verified alerts."""
        Path(self.config.incident_db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.config.incident_db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS verified_incidents (
                    subject_id TEXT PRIMARY KEY,
                    detection_rule TEXT NOT NULL,
                    initial_confidence REAL NOT NULL,
                    peak_confidence REAL NOT NULL,
                    created_at REAL NOT NULL,
                    verified_at REAL NOT NULL,
                    cameras_seen_json TEXT NOT NULL,
                    face_crop_b64 TEXT,
                    frames_json TEXT NOT NULL
                )
                """
            )

    def set_detection_rule(
        self,
        detection_rule: str,
        threshold: float | None = None,
        timeout_sec: float | None = None,
    ) -> dict[str, Any]:
        """Updates the active zero-shot detection rule and optional threshold/TTL from the Client UI."""
        with self._lock:
            rule_clean = detection_rule.strip()
            if rule_clean:
                self.current_detection_rule = rule_clean
            if threshold is not None:
                self.decision_threshold = max(0.05, min(0.99, float(threshold)))
                self.config.decision_threshold = self.decision_threshold
            if timeout_sec is not None:
                self.verification_timeout_sec = max(5.0, min(600.0, float(timeout_sec)))
                self.config.verification_timeout_sec = self.verification_timeout_sec

            payload = {
                "type": "rule_updated",
                "detection_rule": self.current_detection_rule,
                "threshold": self.decision_threshold,
                "verification_timeout_sec": self.verification_timeout_sec,
                "timestamp": time.time(),
            }
        self._emit(payload)
        return payload

    def resolve_existing_subject_for_track(self, camera_id: str, track_id: int) -> FlaggedSubjectRecord | None:
        """Checks if `(camera_id, track_id)` is already associated with an active flagged subject."""
        with self._lock:
            sid = self.track_to_subject.get((camera_id, track_id))
            if not sid:
                return None
            rec = self.flagged_subjects.get(sid)
            if rec is None or rec.status in ("dismissed", "expired"):
                self.track_to_subject.pop((camera_id, track_id), None)
                return None
            return rec

    def register_or_update_subject(
        self,
        camera_id: str,
        track_id: int,
        bbox: BoundingBox,
        match_probability: float,
        padded_crop_pil: Image.Image,
        full_frame_bgr: np.ndarray,
        face_embedding: np.ndarray | None,
        face_pil: Image.Image | None,
    ) -> tuple[FlaggedSubjectRecord, bool, float | None]:
        """Handles a positive detection rule match or cross-camera biometric match:
          1. Checks if `(camera_id, track_id)` is already linked to a flagged subject.
          2. Else if `face_embedding` is present, queries LanceDB to see if this subject
             was already flagged on this or another camera (cross-camera Re-ID).
          3. Else creates a new `unverified` subject record with a verification TTL
             and stores their facial embedding in LanceDB.
          4. Captures up to 10 incident frames during the detection window, drawing a
             thin bounding box on both the full scene frame and padded crop.

        Returns:
            (FlaggedSubjectRecord, is_new_detection, reid_similarity)
        """
        now = time.time()
        reid_sim: float | None = None
        is_new = False
        img_h, img_w = full_frame_bgr.shape[:2]

        with self._lock:
            rec = self.resolve_existing_subject_for_track(camera_id, track_id)

            # Cross-camera Re-ID via LanceDB biometric vector search
            if rec is None and face_embedding is not None:
                matched_row, sim = self.vector_db.search_matching_subject(face_embedding, now=now)
                if matched_row is not None:
                    matched_sid = matched_row["subject_id"]
                    rec = self.flagged_subjects.get(matched_sid)
                    reid_sim = sim
                    if rec is not None:
                        self.track_to_subject[(camera_id, track_id)] = rec.subject_id
                        rec.active_tracks[camera_id] = track_id
                        if camera_id not in rec.cameras_seen:
                            rec.cameras_seen.append(camera_id)

            # New Unverified Flagged Subject
            if rec is None:
                is_new = True
                sid = f"SUBJ-{uuid.uuid4().hex[:6].upper()}"
                expires_at = now + self.verification_timeout_sec
                face_b64 = pil_to_b64_jpeg(face_pil, quality=85) if face_pil is not None else None

                rec = FlaggedSubjectRecord(
                    subject_id=sid,
                    status="unverified",
                    detection_rule=self.current_detection_rule,
                    initial_confidence=match_probability,
                    peak_confidence=match_probability,
                    created_at=now,
                    expires_at=expires_at,
                    cameras_seen=[camera_id],
                    active_tracks={camera_id: track_id},
                    latest_bbox=bbox.to_dict(),
                    frame_width=img_w,
                    frame_height=img_h,
                    has_face_embedding=face_embedding is not None,
                    face_crop_b64=face_b64,
                )
                self.flagged_subjects[sid] = rec
                self.track_to_subject[(camera_id, track_id)] = sid

                if face_embedding is not None:
                    self.vector_db.upsert_subject_embedding(
                        subject_id=sid,
                        embedding=face_embedding,
                        status="unverified",
                        detection_rule=self.current_detection_rule,
                        confidence=match_probability,
                        created_at=now,
                        expires_at=expires_at,
                        camera_id=camera_id,
                        track_id=track_id,
                    )
            else:
                # Update existing flagged subject metadata and live coordinates
                rec.peak_confidence = max(rec.peak_confidence, match_probability)
                rec.active_tracks[camera_id] = track_id
                rec.latest_bbox = bbox.to_dict()
                rec.frame_width = img_w
                rec.frame_height = img_h
                if camera_id not in rec.cameras_seen:
                    rec.cameras_seen.append(camera_id)
                if face_pil is not None and rec.face_crop_b64 is None:
                    rec.face_crop_b64 = pil_to_b64_jpeg(face_pil, quality=85)
                if face_embedding is not None and not rec.has_face_embedding:
                    rec.has_face_embedding = True
                    self.vector_db.upsert_subject_embedding(
                        subject_id=rec.subject_id,
                        embedding=face_embedding,
                        status=rec.status,
                        detection_rule=rec.detection_rule,
                        confidence=rec.peak_confidence,
                        created_at=rec.created_at,
                        expires_at=rec.expires_at,
                        camera_id=camera_id,
                        track_id=track_id,
                    )

            # Capture up to 10 incident frames during the detection window (with thin bounding boxes)
            last_saved = self._last_frame_saved_at.get(rec.subject_id, 0.0)
            if (is_new or (now - last_saved) >= self.config.frame_capture_interval_sec) and len(rec.frames) < self.config.max_frames_per_subject:
                crop_b64 = render_annotated_crop_b64(
                    padded_crop_pil=padded_crop_pil,
                    bbox=bbox,
                    status=rec.status,
                    quality=self.config.crop_jpeg_quality,
                    max_side=280,
                )
                full_b64 = render_annotated_scene_b64(
                    full_frame_bgr=full_frame_bgr,
                    bbox=bbox,
                    subject_id=rec.subject_id,
                    status=rec.status,
                    match_probability=match_probability,
                    quality=78,
                    max_side=540,
                )
                frame_rec = SubjectFrameRecord(
                    frame_id=f"FRM-{len(rec.frames) + 1:02d}",
                    subject_id=rec.subject_id,
                    camera_id=camera_id,
                    track_id=track_id,
                    timestamp=now,
                    match_probability=match_probability,
                    crop_b64=crop_b64,
                    full_frame_b64=full_b64,
                    bbox=bbox.to_dict(),
                    face_crop_b64=rec.face_crop_b64,
                )
                rec.frames.append(frame_rec)
                self._last_frame_saved_at[rec.subject_id] = now

            event_payload = {
                "type": "subject_flagged" if is_new else "subject_updated",
                "subject": rec.to_dict(include_frames=True),
                "camera_id": camera_id,
                "track_id": track_id,
                "reid_similarity": reid_sim,
                "timestamp": now,
            }

        self._emit(event_payload)
        return rec, is_new, reid_sim

    def verify_subject(self, subject_id: str) -> dict[str, Any] | None:
        """Operator verified an unverified flagged subject:
          - Updates status to 'verified' (turns live bounding box RED)
          - Clears TTL expiration in LanceDB so biometric embedding persists across cameras
          - Archives incident record in SQLite Incident Database
        """
        now = time.time()
        with self._lock:
            rec = self.flagged_subjects.get(subject_id)
            if rec is None or rec.status in ("dismissed", "expired"):
                return None

            rec.status = "verified"
            rec.verified_at = now
            rec.expires_at = None
            self.vector_db.promote_to_verified(subject_id)
            self._persist_to_incident_db(rec)

            payload = {
                "type": "subject_verified",
                "subject": rec.to_dict(include_frames=True),
                "timestamp": now,
            }

        self._emit(payload)
        return payload

    def dismiss_subject(self, subject_id: str) -> dict[str, Any] | None:
        """Operator dismissed a flagged subject alert:
          - Immediately deletes facial embedding from LanceDB
          - Removes subject from active watchlist
        """
        now = time.time()
        with self._lock:
            rec = self.flagged_subjects.pop(subject_id, None)
            if rec is None:
                return None
            rec.status = "dismissed"
            self.vector_db.remove_subject(subject_id)
            for key, sid in list(self.track_to_subject.items()):
                if sid == subject_id:
                    del self.track_to_subject[key]

            payload = {
                "type": "subject_dismissed",
                "subject_id": subject_id,
                "timestamp": now,
            }

        self._emit(payload)
        return payload

    def cleanup_expired_subjects(self) -> list[str]:
        """Checks for unverified flagged subjects whose verification TTL has elapsed:
          - Purges their facial embeddings from LanceDB
          - Removes them from active watchlist and notifies connected clients
        """
        now = time.time()
        expired_ids: list[str] = []

        with self._lock:
            db_purged = set(self.vector_db.purge_expired_unverified_subjects(now=now))

            for sid, rec in list(self.flagged_subjects.items()):
                if rec.status == "unverified" and rec.expires_at is not None and now >= rec.expires_at:
                    rec.status = "expired"
                    self.vector_db.remove_subject(sid)
                    del self.flagged_subjects[sid]
                    expired_ids.append(sid)
                elif sid in db_purged and sid in self.flagged_subjects:
                    del self.flagged_subjects[sid]
                    expired_ids.append(sid)

            if expired_ids:
                for key, sid in list(self.track_to_subject.items()):
                    if sid in expired_ids:
                        del self.track_to_subject[key]

        for sid in expired_ids:
            self._emit(
                {
                    "type": "subject_expired",
                    "subject_id": sid,
                    "reason": "Verification TTL elapsed; facial embedding purged from LanceDB.",
                    "timestamp": now,
                }
            )
        return expired_ids

    def _persist_to_incident_db(self, rec: FlaggedSubjectRecord) -> None:
        with sqlite3.connect(self.config.incident_db_path) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO verified_incidents (
                    subject_id, detection_rule, initial_confidence, peak_confidence,
                    created_at, verified_at, cameras_seen_json, face_crop_b64, frames_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    rec.subject_id,
                    rec.detection_rule,
                    rec.initial_confidence,
                    rec.peak_confidence,
                    rec.created_at,
                    rec.verified_at,
                    json.dumps(rec.cameras_seen),
                    rec.face_crop_b64,
                    json.dumps([f.to_dict() for f in rec.frames]),
                ),
            )

    def get_snapshot_state(self) -> dict[str, Any]:
        """Returns full current system snapshot for newly connected WebSocket / REST clients."""
        with self._lock:
            return {
                "type": "state_snapshot",
                "detection_rule": self.current_detection_rule,
                "threshold": self.decision_threshold,
                "verification_timeout_sec": self.verification_timeout_sec,
                "subjects": [
                    s.to_dict(include_frames=True)
                    for s in sorted(self.flagged_subjects.values(), key=lambda x: x.created_at, reverse=True)
                ],
                "lancedb_embeddings": self.vector_db.list_embeddings_metadata(),
                "timestamp": time.time(),
            }

    def _emit(self, payload: dict[str, Any]) -> None:
        if self.on_event_callback is not None:
            self.on_event_callback(payload)
