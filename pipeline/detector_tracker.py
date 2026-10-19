"""Person Detector (Ultralytics YOLO26s) + Multi-Object Tracker (ByteTrack) + Padded Cropper."""

from __future__ import annotations
from typing import Any
import cv2
import numpy as np
from PIL import Image

from config import PipelineConfig
from pipeline.schemas import BoundingBox


def compute_padded_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    img_w: int,
    img_h: int,
    padding_ratio: float = 0.35,
) -> BoundingBox:
    """Computes a tight bounding box and a context-padded bounding box clamped to image bounds.

    Feeding the VLM a padded crop provides surrounding visual context (e.g., hand gestures,
    smoke plumes, held objects, nearby restricted zones) that a tight person box would clip.
    """
    bx1, by1 = max(0, int(round(x1))), max(0, int(round(y1)))
    bx2, by2 = min(img_w, int(round(x2))), min(img_h, int(round(y2)))

    bw = max(1, bx2 - bx1)
    bh = max(1, by2 - by1)

    pad_x = int(round(bw * padding_ratio))
    pad_y = int(round(bh * padding_ratio))

    px1 = max(0, bx1 - pad_x)
    py1 = max(0, by1 - pad_y)
    px2 = min(img_w, bx2 + pad_x)
    py2 = min(img_h, by2 + pad_y)

    return BoundingBox(
        x1=bx1,
        y1=by1,
        x2=bx2,
        y2=by2,
        padded_x1=px1,
        padded_y1=py1,
        padded_x2=px2,
        padded_y2=py2,
    )


def extract_padded_crop_pil(frame_bgr: np.ndarray, bbox: BoundingBox) -> Image.Image:
    """Extracts the context-padded region from a BGR OpenCV frame and converts it to RGB PIL Image."""
    crop_bgr = frame_bgr[bbox.padded_y1 : bbox.padded_y2, bbox.padded_x1 : bbox.padded_x2]
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(crop_rgb)


class PersonDetectorTracker:
    """Per-camera person detector and ByteTrack tracker using Ultralytics YOLO26s."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self._models_by_camera: dict[str, Any] = {}
        self.device = self._detect_device()

    @staticmethod
    def _detect_device() -> str:
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass
        return "cpu"

    def _get_or_create_tracker_model(self, camera_id: str):
        """Each camera stream gets its own stateful ByteTrack instance via Ultralytics YOLO."""
        if camera_id in self._models_by_camera:
            return self._models_by_camera[camera_id]

        from ultralytics import YOLO

        model = YOLO(self.config.yolo_model_path)
        self._models_by_camera[camera_id] = model
        return model

    def detect_and_track(
        self, camera_id: str, frame_bgr: np.ndarray
    ) -> list[tuple[int, float, BoundingBox, Image.Image, np.ndarray]]:
        """Runs YOLO26s + ByteTrack on a single camera frame.

        Returns a list of tuples:
            (track_id, confidence, BoundingBox, padded_crop_pil, tight_crop_bgr)
        """
        img_h, img_w = frame_bgr.shape[:2]
        detections: list[tuple[int, float, BoundingBox, Image.Image, np.ndarray]] = []

        model = self._get_or_create_tracker_model(camera_id)
        results = model.track(
            source=frame_bgr,
            persist=True,
            tracker=self.config.tracker_config,
            classes=[0],  # COCO class 0 == person
            conf=self.config.person_conf_threshold,
            iou=self.config.person_iou_threshold,
            device=self.device,
            verbose=False,
        )

        boxes = results[0].boxes
        if boxes is None or boxes.id is None:
            # ByteTrack has not assigned track IDs yet
            return detections

        xyxy_arr = boxes.xyxy.cpu().numpy()
        conf_arr = boxes.conf.cpu().numpy()
        id_arr = boxes.id.int().cpu().numpy()

        for (x1, y1, x2, y2), conf, tid in zip(xyxy_arr, conf_arr, id_arr):
            bbox = compute_padded_bbox(
                x1, y1, x2, y2, img_w, img_h, padding_ratio=self.config.crop_padding_ratio
            )
            if (bbox.x2 - bbox.x1) < 20 or (bbox.y2 - bbox.y1) < 30:
                continue
            padded_pil = extract_padded_crop_pil(frame_bgr, bbox)
            tight_bgr = frame_bgr[bbox.y1 : bbox.y2, bbox.x1 : bbox.x2].copy()
            detections.append((int(tid), float(conf), bbox, padded_pil, tight_bgr))

        return detections
