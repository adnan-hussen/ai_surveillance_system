#!/usr/bin/env python3
"""Edge Compute Video Analytics & Security Surveillance Server.

Orchestrates:
  1. Real-time multi-camera video ingestion (RTSP / USB / built-in webcam / video file)
  2. Person Detection & Multi-Object Tracking (Ultralytics YOLO26s + ByteTrack)
  3. Context-padded subject cropping (configurable padding ratio around person bbox)
  4. System-One Multimodal Decision Model (`LiquidAI/d1-3B`) single-pass detection rule evaluation
  5. Face Detection & 512-D Biometric Embedding (`InsightFace buffalo_l`: SCRFD + ArcFace ResNet-50)
  6. Vector Database (`LanceDB`) for unverified/verified subject Re-ID across cameras & TTL purge
  7. Event Service & RFC 6455 WebSocket + HTTP REST API for the React Security Operations Console

Usage:
    python server.py --webcam                 # Stream local MacBook / USB camera at 30 FPS
    python server.py --compile                # Enable CUDA graphs for d1-3B
    python server.py --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import logging
import mimetypes
import struct
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from config import PipelineConfig
from pipeline.decision_model import DecisionModelEngine
from pipeline.detector_tracker import PersonDetectorTracker
from pipeline.event_service import EventService, bgr_to_b64_jpeg
from pipeline.face_embedder import FaceDetectorEmbedder
from pipeline.schemas import TrackedSubject
from pipeline.vector_db import SubjectVectorDB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("surveillance.server")

WEB_DIR = Path(__file__).parent / "web"
MAX_BODY = 4 << 20
WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


# ============================================================================
# RFC 6455 WebSocket Implementation (Zero External Web Framework Dependencies)
# ============================================================================
def _unmask(data: bytes, mask: bytes) -> bytes:
    n = len(data)
    key = int.from_bytes((mask * (n // 4 + 1))[:n], "big")
    return (int.from_bytes(data, "big") ^ key).to_bytes(n, "big")


class WebSocketConnection:
    """RFC 6455 WebSocket connection supporting concurrent broadcasts and client commands."""

    def __init__(self, rfile, wfile):
        self.rfile = rfile
        self.wfile = wfile
        self.lock = threading.Lock()
        self.alive = True

    def _exact(self, n: int) -> bytes:
        data = self.rfile.read(n)
        if len(data) < n:
            raise ConnectionError("WebSocket closed")
        return data

    def _frame(self) -> tuple[int, bool, bytes]:
        b0, b1 = self._exact(2)
        n = b1 & 0x7F
        if n == 126:
            n = struct.unpack(">H", self._exact(2))[0]
        elif n == 127:
            n = struct.unpack(">Q", self._exact(8))[0]
        if n > MAX_BODY:
            raise ConnectionError("WebSocket frame too large")
        mask = self._exact(4) if (b1 & 0x80) else b""
        data = self._exact(n)
        if mask:
            data = _unmask(data, mask)
        return b0 & 0x0F, bool(b0 & 0x80), data

    def receive(self) -> str | None:
        parts: list[bytes] = []
        while self.alive:
            op, fin, data = self._frame()
            if op == 0x8:
                self.send(data[:2], 0x8)
                self.alive = False
                return None
            if op == 0x9:
                self.send(data, 0xA)
                continue
            if op in (0x1, 0x0):
                parts.append(data)
                if sum(map(len, parts)) > MAX_BODY:
                    raise ConnectionError("Message too large")
                if fin:
                    return b"".join(parts).decode("utf-8", errors="replace")
        return None

    def send(self, data: bytes | str, op: int = 0x1) -> None:
        if not self.alive:
            return
        if isinstance(data, str):
            data = data.encode("utf-8")
        n = len(data)
        head = bytes([0x80 | op]) + (
            bytes([n])
            if n < 126
            else bytes([126]) + struct.pack(">H", n)
            if n < 65536
            else bytes([127]) + struct.pack(">Q", n)
        )
        with self.lock:
            try:
                self.wfile.write(head + data)
                self.wfile.flush()
            except OSError:
                self.alive = False
                raise


class WebSocketHub:
    """Manages active WebSocket clients for broadcasting live camera frames & security alerts."""

    def __init__(self):
        self._clients: set[WebSocketConnection] = set()
        self._lock = threading.Lock()

    def register(self, ws: WebSocketConnection) -> None:
        with self._lock:
            self._clients.add(ws)

    def unregister(self, ws: WebSocketConnection) -> None:
        with self._lock:
            self._clients.discard(ws)

    def broadcast_json(self, payload: dict[str, Any]) -> None:
        raw = json.dumps(payload)
        with self._lock:
            clients = list(self._clients)
        dead: list[WebSocketConnection] = []
        for ws in clients:
            try:
                ws.send(raw)
            except Exception:
                dead.append(ws)
        if dead:
            with self._lock:
                for d in dead:
                    self._clients.discard(d)


# ============================================================================
# Master Multi-Camera Video Analytics Pipeline Orchestrator
# ============================================================================
class SurveillancePipelineOrchestrator:
    """Runs the end-to-end Edge Compute pipeline across all configured camera streams."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.ws_hub = WebSocketHub()
        self.vector_db = SubjectVectorDB(config)
        self.event_service = EventService(
            config=config,
            vector_db=self.vector_db,
            on_event_callback=self.ws_hub.broadcast_json,
        )
        self.detector_tracker = PersonDetectorTracker(config)
        self.decision_model = DecisionModelEngine(config)
        self.face_embedder = FaceDetectorEmbedder(config)

        self._running = False
        self._threads: list[threading.Thread] = []
        self._last_eval_time: dict[tuple[str, int], float] = {}
        self._last_eval_result: dict[tuple[str, int], dict[str, Any]] = {}
        self._last_latency_ms = 0.0

    def start(self) -> None:
        """Loads models and starts camera processing threads + LanceDB TTL reaper thread."""
        self.decision_model.load()
        self.face_embedder.load()
        self._running = True

        # Start TTL cleanup thread for expiring unverified subjects in LanceDB
        reaper = threading.Thread(target=self._ttl_reaper_loop, daemon=True, name="lancedb_ttl_reaper")
        reaper.start()
        self._threads.append(reaper)

        # Start one processing thread per camera stream
        for cam_id, source in self.config.camera_sources.items():
            t = threading.Thread(
                target=self._camera_loop,
                args=(cam_id, source),
                daemon=True,
                name=f"cam_{cam_id}",
            )
            t.start()
            self._threads.append(t)

    def stop(self) -> None:
        self._running = False

    def _ttl_reaper_loop(self) -> None:
        while self._running:
            try:
                self.event_service.cleanup_expired_subjects()
            except Exception as exc:
                logger.error("TTL reaper error: %s", exc)
            time.sleep(self.config.ttl_cleanup_interval_sec)

    def _camera_loop(self, camera_id: str, source: str | int) -> None:
        """High-FPS asynchronous camera capture and streaming loop.

        Runs video capture and streaming at 30 FPS. Detection and VLM evaluation
        are scheduled asynchronously so camera frames never block waiting on inference.
        """
        cap = cv2.VideoCapture(source)
        if isinstance(source, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)

        frame_interval = 1.0 / max(1, self.config.target_fps)
        infer_in_flight = False
        latest_tracks: list[dict[str, Any]] = []
        frame_idx = 0

        infer_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix=f"infer_{camera_id}")

        def run_infer(f_bgr: np.ndarray) -> None:
            nonlocal latest_tracks, infer_in_flight
            try:
                latest_tracks = self._evaluate_frame_subjects(camera_id, f_bgr)
            finally:
                infer_in_flight = False

        while self._running:
            t_start = time.perf_counter()
            ok, frame_bgr = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                time.sleep(0.05)
                continue

            frame_idx += 1
            if not infer_in_flight and (frame_idx % 2 == 0):
                infer_in_flight = True
                infer_pool.submit(run_infer, frame_bgr.copy())

            self._broadcast_high_fps_stream(camera_id, frame_bgr, latest_tracks)

            elapsed = time.perf_counter() - t_start
            sleep_for = max(0.005, frame_interval - elapsed)
            time.sleep(sleep_for)

        cap.release()
        infer_pool.shutdown(wait=False)

    def _evaluate_frame_subjects(self, camera_id: str, frame_bgr: np.ndarray) -> list[dict[str, Any]]:
        """Asynchronous worker: executes YOLO26s + ByteTrack + d1-3B System-One on keyframes."""
        img_h, img_w = frame_bgr.shape[:2]
        now = time.time()
        raw_tracks = self.detector_tracker.detect_and_track(camera_id, frame_bgr)
        out_tracks: list[dict[str, Any]] = []

        for track_id, det_conf, bbox, padded_pil, tight_bgr in raw_tracks:
            key = (camera_id, track_id)
            last_eval = self._last_eval_time.get(key, 0.0)

            if (now - last_eval) >= self.config.decision_eval_interval_sec or key not in self._last_eval_result:
                decision_out = self.decision_model.evaluate_subject_crop(
                    padded_crop=padded_pil,
                    detection_rule=self.event_service.current_detection_rule,
                    camera_id=camera_id,
                    track_id=track_id,
                )
                self._last_eval_time[key] = now
                self._last_eval_result[key] = decision_out
                self._last_latency_ms = decision_out["latency_ms"]
            else:
                decision_out = self._last_eval_result[key]

            match_prob = float(decision_out["match_probability"])
            is_triggered = bool(decision_out["is_triggered"])
            existing_subject = self.event_service.resolve_existing_subject_for_track(camera_id, track_id)
            subject_rec = existing_subject
            reid_sim = None
            face_detected = False

            if is_triggered or existing_subject is not None:
                face_emb, face_pil, _ = self.face_embedder.detect_and_embed(tight_bgr)
                face_detected = face_emb is not None
                subject_rec, _, reid_sim = self.event_service.register_or_update_subject(
                    camera_id=camera_id,
                    track_id=track_id,
                    bbox=bbox,
                    match_probability=match_prob,
                    padded_crop_pil=padded_pil,
                    full_frame_bgr=frame_bgr,
                    face_embedding=face_emb,
                    face_pil=face_pil,
                )
            elif len(self.event_service.flagged_subjects) > 0:
                # Cross-camera Re-ID check against active watchlist in LanceDB
                face_emb, face_pil, _ = self.face_embedder.detect_and_embed(tight_bgr)
                if face_emb is not None:
                    matched_row, sim = self.vector_db.search_matching_subject(face_emb, now=now)
                    if matched_row is not None:
                        face_detected = True
                        subject_rec, _, reid_sim = self.event_service.register_or_update_subject(
                            camera_id=camera_id,
                            track_id=track_id,
                            bbox=bbox,
                            match_probability=match_prob,
                            padded_crop_pil=padded_pil,
                            full_frame_bgr=frame_bgr,
                            face_embedding=face_emb,
                            face_pil=face_pil,
                        )

            ts = TrackedSubject(
                camera_id=camera_id,
                track_id=track_id,
                bbox=bbox,
                det_confidence=det_conf,
                frame_width=img_w,
                frame_height=img_h,
                subject_id=subject_rec.subject_id if subject_rec else None,
                subject_status=subject_rec.status if subject_rec else None,
                match_probability=match_prob,
                detection_rule=self.event_service.current_detection_rule,
                face_detected=face_detected,
                reid_similarity=reid_sim,
            )
            out_tracks.append(ts.to_dict())
        return out_tracks

    def _broadcast_high_fps_stream(
        self, camera_id: str, frame_bgr: np.ndarray, tracks: list[dict[str, Any]]
    ) -> None:
        """Broadcasts clean video frames + exact track bounding box coordinates at 30 FPS.

        Bounding boxes are rendered exclusively by the client's 60 FPS hardware overlay
        canvas using the exact `tracks` coordinates so boxes are never duplicated and
        reflect status transitions (Yellow -> Red) and crop-padding toggles immediately.
        """
        orig_h, orig_w = frame_bgr.shape[:2]

        # Refresh each track's current subject status from EventService in case operator just verified/dismissed
        synced_tracks: list[dict[str, Any]] = []
        for t in tracks:
            t_copy = dict(t)
            sid = t_copy.get("subject_id")
            if sid:
                rec = self.event_service.flagged_subjects.get(sid)
                if rec is not None and rec.status in ("unverified", "verified"):
                    t_copy["subject_status"] = rec.status
                else:
                    t_copy["subject_id"] = None
                    t_copy["subject_status"] = None
            synced_tracks.append(t_copy)

        stream_bgr = frame_bgr
        if orig_w > 640:
            scale = 640.0 / orig_w
            stream_bgr = cv2.resize(frame_bgr, (640, int(orig_h * scale)), interpolation=cv2.INTER_LINEAR)

        self.ws_hub.broadcast_json(
            {
                "type": "camera_frame",
                "camera_id": camera_id,
                "frame_b64": bgr_to_b64_jpeg(stream_bgr, quality=65),
                "frame_width": orig_w,
                "frame_height": orig_h,
                "tracks": synced_tracks,
                "decision_latency_ms": self._last_latency_ms,
                "detection_rule": self.event_service.current_detection_rule,
                "threshold": self.event_service.decision_threshold,
                "timestamp": time.time(),
            }
        )


# ============================================================================
# HTTP + WebSocket Request Handler
# ============================================================================
def make_http_handler(orchestrator: SurveillancePipelineOrchestrator):
    class SurveillanceHandler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        disable_nagle_algorithm = True

        def log_message(self, *args):
            pass

        def _send_json(self, code: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload).encode("utf-8")
            self._send_bytes(code, body, "application/json; charset=utf-8")

        def _send_bytes(self, code: int, body: bytes, content_type: str) -> None:
            try:
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("Access-Control-Allow-Origin", "*")
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass

        def do_OPTIONS(self):
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
            self.send_header("Access-Control-Allow-Headers", "Content-Type")
            self.end_headers()

        def do_GET(self):
            route = self.path.split("?")[0]
            if route == "/ws":
                self._handle_websocket()
                return
            if route == "/api/state":
                self._send_json(200, orchestrator.event_service.get_snapshot_state())
                return

            name = route.lstrip("/") or "index.html"
            file_path = (WEB_DIR / name).resolve()
            if WEB_DIR.resolve() not in file_path.parents or not file_path.is_file():
                self._send_bytes(404, b"Not Found", "text/plain")
                return
            mime = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
            if mime in ("text/javascript", "application/javascript"):
                mime = "text/javascript; charset=utf-8"
            self._send_bytes(200, file_path.read_bytes(), mime)

        def do_POST(self):
            route = self.path.split("?")[0]
            size = int(self.headers.get("Content-Length", 0))
            if size > MAX_BODY:
                self._send_json(413, {"error": "Payload too large"})
                return
            try:
                body = json.loads(self.rfile.read(size)) if size > 0 else {}
            except Exception:
                self._send_json(400, {"error": "Invalid JSON"})
                return

            if route == "/api/rule":
                rule_text = str(body.get("detection_rule") or "")
                res = orchestrator.event_service.set_detection_rule(
                    detection_rule=rule_text,
                    threshold=body.get("threshold"),
                    timeout_sec=body.get("verification_timeout_sec"),
                )
                self._send_json(200, res)
                return

            if route == "/api/verify":
                sid = str(body.get("subject_id") or "")
                if str(body.get("decision", "")).lower() == "verify":
                    res = orchestrator.event_service.verify_subject(sid)
                else:
                    res = orchestrator.event_service.dismiss_subject(sid)
                if res is None:
                    self._send_json(404, {"error": f"Subject {sid} not found or already resolved"})
                else:
                    self._send_json(200, res)
                return

            self._send_json(404, {"error": "Unknown endpoint"})

        def _handle_websocket(self):
            key = self.headers.get("Sec-WebSocket-Key")
            if "websocket" not in self.headers.get("Upgrade", "").lower() or not key:
                self._send_bytes(400, b"Bad WebSocket handshake", "text/plain")
                self.close_connection = True
                return

            accept = base64.b64encode(hashlib.sha1((key + WS_GUID).encode()).digest()).decode()
            self.send_response(101)
            self.send_header("Upgrade", "websocket")
            self.send_header("Connection", "Upgrade")
            self.send_header("Sec-WebSocket-Accept", accept)
            self.end_headers()
            self.close_connection = True

            ws = WebSocketConnection(self.rfile, self.wfile)
            orchestrator.ws_hub.register(ws)
            try:
                ws.send(json.dumps(orchestrator.event_service.get_snapshot_state()))
                while (msg_text := ws.receive()) is not None:
                    try:
                        cmd = json.loads(msg_text)
                        action = cmd.get("action")
                        if action == "set_detection_rule":
                            rule_text = str(cmd.get("detection_rule") or "")
                            orchestrator.event_service.set_detection_rule(
                                detection_rule=rule_text,
                                threshold=cmd.get("threshold"),
                                timeout_sec=cmd.get("verification_timeout_sec"),
                            )
                        elif action == "verify_subject":
                            orchestrator.event_service.verify_subject(str(cmd.get("subject_id") or ""))
                        elif action == "dismiss_subject":
                            orchestrator.event_service.dismiss_subject(str(cmd.get("subject_id") or ""))
                    except Exception as exc:
                        ws.send(json.dumps({"type": "error", "message": str(exc)}))
            except (ConnectionError, OSError):
                pass
            finally:
                orchestrator.ws_hub.unregister(ws)

    return SurveillanceHandler


class ThreadedServer(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        if not isinstance(sys.exc_info()[1], (BrokenPipeError, ConnectionResetError)):
            super().handle_error(request, client_address)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="LiquidAI/d1-3B", help="System-One VLM repository ID")
    parser.add_argument("--yolo-model", default="yolo26s.pt", help="Ultralytics YOLO26 weights")
    parser.add_argument("--compile", action="store_true", help="Compile d1-3B with CUDA graphs")
    parser.add_argument("--padding-ratio", type=float, default=0.35, help="Context padding ratio around subject bbox")
    parser.add_argument("--threshold", type=float, default=0.65, help="Positive match probability threshold")
    parser.add_argument("--timeout", type=float, default=45.0, help="Seconds before unverified subject expires from LanceDB")
    parser.add_argument("--source", type=str, default = "0", help="Select camera souce")
    args = parser.parse_args()

    cfg = PipelineConfig(
        yolo_model_path=args.yolo_model,
        decision_model_id=args.model,
        compile_decision_model=args.compile,
        crop_padding_ratio=args.padding_ratio,
        decision_threshold=args.threshold,
        verification_timeout_sec=args.timeout,
    )
    if args.source == "0":
        cfg.camera_sources["CAM-01 (Primary Feed)"] = 0
    else:
        cfg.camera_sources["CAM-01 (Primary Feed)"] = f"{args.source}"

    orchestrator = SurveillancePipelineOrchestrator(cfg)
    orchestrator.start()

    server = ThreadedServer((args.host, args.port), make_http_handler(orchestrator))
    logger.info("Surveillance Server ready at http://%s:%d/", args.host, args.port)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        orchestrator.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
