"""Face Detection (InsightFace SCRFD) and Biometric Embedding (ArcFace ResNet-50) via the buffalo_l model pack."""

from __future__ import annotations
import cv2
import numpy as np
from PIL import Image

from config import PipelineConfig

# Standard 112x112 5-point reference landmarks for ArcFace alignment
REFERENCE_FACIAL_POINTS_112 = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)


def align_face_112(img_bgr: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """Aligns a detected face to standard 112x112 using 5-point facial landmarks."""
    tform, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), REFERENCE_FACIAL_POINTS_112, method=cv2.LMEDS)
    return cv2.warpAffine(img_bgr, tform, (112, 112), borderValue=0.0)


class FaceDetectorEmbedder:
    """InsightFace buffalo_l face detector (SCRFD) and 512-D ArcFace (ResNet-50) recognition model."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.app = None

    def load(self) -> None:
        """Initializes the InsightFace buffalo_l model pack with detection & recognition modules."""
        import torch
        from insightface.app import FaceAnalysis

        if torch.cuda.is_available():
            device = "cuda"
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]
        else:
            device = "cpu"
            providers = ["CPUExecutionProvider"]

        self.app = FaceAnalysis(
            name=self.config.face_model_pack,
            allowed_modules=["detection", "recognition"],
            providers=providers,
        )
        self.app.prepare(
            ctx_id=0 if device == "cuda" else -1,
            det_thresh=self.config.face_det_threshold,
            det_size=self.config.face_det_size,
        )

    def detect_and_embed(
        self, person_crop_bgr: np.ndarray
    ) -> tuple[np.ndarray | None, Image.Image | None, float]:
        """Detects the most prominent face in the person crop and computes its 512-D ArcFace vector.

        Returns:
            (embedding_512_float32, aligned_face_pil, face_det_score) or (None, None, 0.0) if no face is visible.
        """
        faces = self.app.get(person_crop_bgr)
        if not faces:
            return None, None, 0.0

        # Select largest & highest-confidence face in the person crop
        face = max(
            faces,
            key=lambda f: float((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])) * float(f.det_score),
        )

        aligned_bgr = align_face_112(person_crop_bgr, face.kps)
        embedding = face.embedding.astype(np.float32)
        embedding /= np.linalg.norm(embedding)

        aligned_pil = Image.fromarray(cv2.cvtColor(aligned_bgr, cv2.COLOR_BGR2RGB))
        return embedding, aligned_pil, float(face.det_score)
