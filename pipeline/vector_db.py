"""LanceDB Vector Store Manager for Unverified and Verified Flagged Subject Facial Embeddings."""

from __future__ import annotations
import threading
import time
from pathlib import Path
from typing import Any
import numpy as np

import lancedb
import pyarrow as pa

from config import PipelineConfig


class SubjectVectorDB:
    """Manages facial embeddings in LanceDB with strict privacy & TTL retention policies:

    - ONLY facial embeddings of 'unverified' and 'verified' flagged subjects are stored.
    - Normal passersby are NEVER embedded or stored.
    - 'unverified' subject embeddings carry an `expires_at` timestamp; if the verification
      timeout elapses without operator confirmation, the embedding is automatically purged
      from LanceDB.
    - If the operator dismisses an alert, the embedding is immediately deleted from LanceDB.
    - If the operator verifies a flagged subject, `status` is updated to 'verified' and
      `expires_at` is set to -1 (permanent retention), enabling persistent cross-camera Re-ID.
    """

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._lock = threading.RLock()
        Path(self.config.lancedb_uri).parent.mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(self.config.lancedb_uri)
        schema = pa.schema(
            [
                pa.field("subject_id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(), self.config.embedding_dim)),
                pa.field("status", pa.string()),  # "unverified" | "verified"
                pa.field("detection_rule", pa.string()),
                pa.field("confidence", pa.float32()),
                pa.field("created_at", pa.float64()),
                pa.field("expires_at", pa.float64()),  # -1.0 for verified subjects
                pa.field("last_camera_id", pa.string()),
                pa.field("last_track_id", pa.int64()),
            ]
        )
        if self.config.lancedb_table_name in self._db.table_names():
            self._table = self._db.open_table(self.config.lancedb_table_name)
        else:
            self._table = self._db.create_table(self.config.lancedb_table_name, schema=schema)

    def upsert_subject_embedding(
        self,
        subject_id: str,
        embedding: np.ndarray,
        status: str,
        detection_rule: str,
        confidence: float,
        created_at: float,
        expires_at: float | None,
        camera_id: str,
        track_id: int,
    ) -> None:
        """Inserts or updates an unverified/verified flagged subject's facial embedding in LanceDB."""
        if status not in ("unverified", "verified"):
            raise ValueError(f"Only 'unverified' or 'verified' subjects may be stored (got {status!r}).")

        vec_list = embedding.astype(np.float32).tolist()
        exp_val = -1.0 if (status == "verified" or expires_at is None) else float(expires_at)

        row = {
            "subject_id": str(subject_id),
            "vector": vec_list,
            "status": str(status),
            "detection_rule": str(detection_rule),
            "confidence": float(confidence),
            "created_at": float(created_at),
            "expires_at": exp_val,
            "last_camera_id": str(camera_id),
            "last_track_id": int(track_id),
        }

        with self._lock:
            self._table.delete(f"subject_id = '{subject_id}'")
            self._table.add([row])

    def search_matching_subject(
        self, query_embedding: np.ndarray, now: float | None = None
    ) -> tuple[dict[str, Any] | None, float]:
        """Searches LanceDB for the closest active ('unverified' or 'verified') subject face.

        Returns:
            (matched_row_dict, cosine_similarity) or (None, 0.0) if no match exceeds threshold.
        """
        now = now or time.time()
        with self._lock:
            results = (
                self._table.search(query_embedding.astype(np.float32).tolist())
                .metric("cosine")
                .limit(5)
                .to_list()
            )

            for r in results:
                if r["status"] == "unverified" and 0 < r["expires_at"] <= now:
                    continue
                sim = 1.0 - float(r["_distance"])
                threshold = getattr(self.config, "reid_match_cosine_threshold", self.config.face_match_cosine_threshold)
                if sim >= threshold:
                    return r, sim

        return None, 0.0

    def promote_to_verified(self, subject_id: str) -> bool:
        """Marks an unverified flagged subject as 'verified' in LanceDB and removes TTL expiration."""
        with self._lock:
            rows = self._table.search().where(f"subject_id = '{subject_id}'").limit(1).to_list()
            if not rows:
                return False
            row = rows[0]
            row.pop("_distance", None)
            row["status"] = "verified"
            row["expires_at"] = -1.0
            self._table.delete(f"subject_id = '{subject_id}'")
            self._table.add([row])
            return True

    def remove_subject(self, subject_id: str) -> bool:
        """Deletes a subject's facial embedding from LanceDB (on alert dismissal or TTL timeout)."""
        with self._lock:
            self._table.delete(f"subject_id = '{subject_id}'")
            return True

    def purge_expired_unverified_subjects(self, now: float | None = None) -> list[str]:
        """Removes all 'unverified' subject embeddings whose verification timeout has passed.

        Returns list of purged subject_ids.
        """
        now = now or time.time()
        with self._lock:
            expired_filter = f"status = 'unverified' AND expires_at > 0 AND expires_at <= {now}"
            expired_rows = self._table.search().where(expired_filter).limit(100).to_list()
            purged_ids = [r["subject_id"] for r in expired_rows]
            if purged_ids:
                self._table.delete(expired_filter)

        return purged_ids

    def list_embeddings_metadata(self) -> list[dict[str, Any]]:
        """Returns non-vector metadata of all currently stored embeddings in LanceDB."""
        with self._lock:
            rows = self._table.search().limit(200).to_list()

            sanitized = []
            for r in rows:
                vec = r.get("vector", [])
                sanitized.append(
                    {
                        "subject_id": r["subject_id"],
                        "status": r["status"],
                        "detection_rule": r["detection_rule"],
                        "confidence": round(float(r["confidence"]), 4),
                        "created_at": r["created_at"],
                        "expires_at": r["expires_at"] if r["expires_at"] > 0 else None,
                        "last_camera_id": r["last_camera_id"],
                        "last_track_id": r["last_track_id"],
                        "vector_preview": [round(float(x), 3) for x in vec[:6]],
                        "embedding_dim": len(vec),
                    }
                )
            return sanitized
