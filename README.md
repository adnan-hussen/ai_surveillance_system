# Video Analytics & Multi-Camera Security Surveillance Pipeline using pretrained models

Edge Compute and Client-side video analytics pipeline powered by **LiquidAI `d1-3B`** (System-One decision model), **Ultralytics YOLO26s**, **ByteTrack**, **InsightFace `buffalo_l`** (SCRFD-10GF detector + ArcFace ResNet-50 recognizer), and **LanceDB**.

## Architecture & Workflow

1. **Multi-Camera Ingestion & Subject Tracking (`Ultralytics YOLO26s` + `ByteTrack`)**:
   - Reads real-time streams from security cameras and tracks subjects with stateful per-camera `ByteTrack` trackers (`track_id`).
   - Computes a **context-padded bounding box** (default `+35%` padding around the tight subject bounding box) so the decision model observes surrounding context (e.g., hand gestures, smoke plumes, held objects, PPE compliance, or restricted items).
2. **System-One Multimodal Decision Model (`LiquidAI/d1-3B`)**:
   - Evaluates the padded subject crop against the active **Detection Rule** (e.g., *"Is the person in the frame smoking?"*) using `d1-3B` pretrained model, requiring no further training.
3. **Selective Biometric Detection & ArcFace Recognition (`InsightFace buffalo_l`)**:
   - When `match_probability >= threshold` (or when matching against active watchlist subjects), the subject crop is processed by **InsightFace `buffalo_l`**.
   - Landmark Detection: **SCRFD-10GF** detects 5-point facial keypoints and aligns the face crop to `112x112`.
   - Feature Extraction: **ArcFace ResNet-50** (`w600k_r50.onnx`) extracts a standardized 512-dimensional L2-normalized vector with 91.25% multi-racial recognition accuracy.
4. **Privacy-Preserving Vector Database (`LanceDB`) & Cross-Camera Re-ID**:
   - **Only facial embeddings of `unverified` and `verified` flagged subjects are stored** in LanceDB (`subject_embeddings`).
   - Each `unverified` subject record is assigned a `verification_timeout_sec` TTL. A background reaper thread automatically purges unverified embeddings from LanceDB as soon as the verification window elapses.
   - Across all camera streams, detected faces are queried against LanceDB via cosine similarity so flagged subjects are seamlessly re-identified and tracked across cameras.
5. **Security Event Service & React Operations Console (`web/App.jsx` & `web/index.html`)**:
   - Captures a **10-frame incident sequence** during the detection window alongside the live 30–60 FPS video stream.
   - Outlines **Unverified Flagged Subjects** in **Yellow** (`#eab308`) on the live stream, turning **Red** (`#ef4444`) immediately upon operator verification.
   - Clicking **Verify Subject** promotes the subject to `verified` in LanceDB (locking the vector permanently) and archives the 10-frame incident package in the SQLite **Incident Archive Database** (`verified_incidents.db`).
   - Clicking **Dismiss Alert** immediately deletes the biometric embedding from LanceDB and removes the subject from the live watchlist.

## File Structure

```text
zero_shot_surveillance/
├── README.md                      # Architecture, installation, and run instructions
├── requirements.txt               # Python dependencies for all pipeline models & LanceDB
├── requirements-mac.txt           # Apple Silicon (M1/M2/M3/M4) MPS dependencies
├── requirements-nvidia.txt        # PyTorch CUDA 12.6 + requirements.txt
├── Dockerfile                     # Containerized GPU deployment
├── config.py                      # Central configuration (models, padding ratio, thresholds, TTL)
├── server.py                      # Multi-camera orchestrator, HTTP REST API & RFC 6455 WebSocket server
├── pipeline/
│   ├── __init__.py
│   ├── schemas.py                 # Dataclasses for BoundingBox, TrackedSubject, FlaggedSubjectRecord
│   ├── detector_tracker.py        # YOLO26s + ByteTrack per-camera tracker & context-padded cropper
│   ├── decision_model.py          # LiquidAI/d1-3B System-One engine wrapper
│   ├── face_embedder.py           # InsightFace buffalo_l (SCRFD-10GF + ArcFace ResNet-50 512-D embedder)
│   ├── vector_db.py               # SubjectVectorDB LanceDB manager with automatic TTL purge
│   └── event_service.py           # Security Event Service, Watchlist, Re-ID & SQLite Incident DB
└── web/
    ├── index.html                 # Standalone React Security Operations Console
    └── App.jsx                    # React component source (tactical dark UI)
```

## Running on macOS (Apple Silicon) or Linux/NVIDIA

```bash
# Run with local MacBook FaceTime HD / USB camera at http://localhost:8000
python server.py --compile --host 127.0.0.1 --port 8000
```
