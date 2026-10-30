"""Person Re-Identification Feature Extractor using OSNet (Omni-Scale Network).

Provides single-stage 512-dimensional L2-normalized biometric embeddings directly from
YOLO tight person bounding boxes, replacing multi-stage facial recognition.
"""

from __future__ import annotations

import logging
import os
import urllib.request
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from config import PipelineConfig

logger = logging.getLogger("surveillance.reid")

# Standard Re-ID normalization on 256x128 aspect ratio
REID_TRANSFORMS = T.Compose([
    T.Resize((256, 128), interpolation=T.InterpolationMode.BILINEAR),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

OFFICIAL_OSNET_URLS = {
    "osnet_x1_0_msmt17": "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_msmt17.pth",
    "osnet_x1_0_market1501": "https://github.com/KaiyangZhou/deep-person-reid/releases/download/v1.0.0/osnet_x1_0_market1501.pth",
}


# ============================================================================
# OSNet (Omni-Scale Network) Pure-PyTorch Architecture
# Reference: Zhou et al., "Omni-Scale Feature Learning for Person Re-Identification", ICCV 2019
# ============================================================================
class ConvLayer(nn.Module):
    def __init__(self, in_c: int, out_c: int, k: int, s: int = 1, p: int = 0, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, k, stride=s, padding=p, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.bn(self.conv(x)))


class Conv1x1Linear(nn.Module):
    def __init__(self, in_c: int, out_c: int, s: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, stride=s, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.conv(x))


class LightConv3x3(nn.Module):
    """Depthwise-separable 3x3 light convolution block matching official OSNet."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, groups=out_c, bias=False)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        return self.relu(x)


class ChannelGate(nn.Module):
    """Dynamic channel attention gate aggregating multi-scale receptive field streams."""
    def __init__(self, c: int, reduction: int = 16):
        super().__init__()
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(c, max(1, c // reduction), 1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(max(1, c // reduction), c, 1, bias=True)
        self.gate_activation = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input = x
        x = self.global_avgpool(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.gate_activation(x)
        return input * x


class OSBlock(nn.Module):
    """Omni-scale block with 4 parallel light streams capturing multiple scales."""
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        mid_c = out_c // 4
        self.conv1 = ConvLayer(in_c, mid_c, 1)
        self.conv2a = LightConv3x3(mid_c, mid_c)
        self.conv2b = nn.Sequential(LightConv3x3(mid_c, mid_c), LightConv3x3(mid_c, mid_c))
        self.conv2c = nn.Sequential(
            LightConv3x3(mid_c, mid_c), LightConv3x3(mid_c, mid_c), LightConv3x3(mid_c, mid_c)
        )
        self.conv2d = nn.Sequential(
            LightConv3x3(mid_c, mid_c),
            LightConv3x3(mid_c, mid_c),
            LightConv3x3(mid_c, mid_c),
            LightConv3x3(mid_c, mid_c),
        )
        self.gate = ChannelGate(mid_c)
        self.conv3 = Conv1x1Linear(mid_c, out_c)
        self.downsample = Conv1x1Linear(in_c, out_c) if in_c != out_c else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.downsample(x) if self.downsample else x
        x1 = self.conv1(x)
        out = self.conv2a(x1) + self.conv2b(x1) + self.conv2c(x1) + self.conv2d(x1)
        out = self.gate(out)
        out = self.conv3(out)
        return F.relu(out + residual, inplace=True)


class OSNet(nn.Module):
    """Omni-Scale Feature Extractor backbone outputting 512-D L2-normalized vectors."""
    def __init__(
        self,
        blocks: list[int] = [2, 2, 2],
        channels: list[int] = [64, 256, 384, 512],
        feature_dim: int = 512,
    ):
        super().__init__()
        self.conv1 = ConvLayer(3, channels[0], 7, s=2, p=3)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)

        # Stage 2 + transition
        self.conv2 = nn.Sequential(
            OSBlock(channels[0], channels[1]),
            OSBlock(channels[1], channels[1]),
            nn.Sequential(ConvLayer(channels[1], channels[1], 1), nn.AvgPool2d(2, stride=2)),
        )

        # Stage 3 + transition
        self.conv3 = nn.Sequential(
            OSBlock(channels[1], channels[2]),
            OSBlock(channels[2], channels[2]),
            nn.Sequential(ConvLayer(channels[2], channels[2], 1), nn.AvgPool2d(2, stride=2)),
        )

        # Stage 4
        self.conv4 = nn.Sequential(
            OSBlock(channels[2], channels[3]),
            OSBlock(channels[3], channels[3]),
        )

        self.conv5 = ConvLayer(channels[3], channels[3], 1)
        self.global_avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels[3], feature_dim, bias=True),
            nn.BatchNorm1d(feature_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.conv1(x))
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        v = self.global_avgpool(x).flatten(1)
        feat = self.fc(v)
        return F.normalize(feat, p=2, dim=1)


# ============================================================================
# Person Re-ID Embedder Class
# ============================================================================
class PersonReIDEmbedder:
    """Full-body Person Re-ID embedding engine producing 512-D vectors for LanceDB."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.device = self._detect_device()
        self.model: OSNet | None = None

    @staticmethod
    def _detect_device() -> str:
        env_dev = os.getenv("REID_DEVICE")
        if env_dev:
            return env_dev
        if torch.cuda.is_available():
            return "cuda"
        # On Apple Silicon (MPS), keep OSNet on CPU to avoid Metal command buffer collisions with d1-3B.
        return "cpu"

    def load(self) -> None:
        """Initializes the OSNet architecture and loads pretrained weights."""
        logger.info("Initializing OSNet (%s) on %s...", self.config.reid_model_name, self.device)
        self.model = OSNet(feature_dim=self.config.embedding_dim)

        weights_path = Path(self.config.reid_weights_path)
        if not weights_path.is_file():
            # Check fallback directory
            fallback_path = Path(__file__).parent.parent / "data" / "models" / "osnet_x1_0_msmt17.pt"
            if fallback_path.is_file():
                weights_path = fallback_path

        if weights_path.is_file() and weights_path.stat().st_size > 1000:
            try:
                try:
                    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=True)
                except Exception:
                    state_dict = torch.load(str(weights_path), map_location="cpu", weights_only=False)
                if isinstance(state_dict, dict) and "state_dict" in state_dict:
                    state_dict = state_dict["state_dict"]
                # Strip 'module.' prefix if saved with DataParallel
                cleaned_state = {}
                for k, v in state_dict.items():
                    clean_k = k.replace("module.", "").replace("backbone.", "")
                    cleaned_state[clean_k] = v
                msg = self.model.load_state_dict(cleaned_state, strict=False)
                logger.info("Loaded OSNet weights from %s (%s).", weights_path, msg)
            except Exception as exc:
                logger.warning("Could not load weights from %s: %s. Using initialized model.", weights_path, exc)
        else:
            logger.info("No local OSNet weights at %s. Attempting download from official release...", weights_path)
            self._try_auto_download(weights_path)

        self.model.to(self.device)
        self.model.eval()
        logger.info("OSNet Person Re-ID Embedder ready (output dim: %d).", self.config.embedding_dim)

    def _try_auto_download(self, dest_path: Path) -> None:
        url = OFFICIAL_OSNET_URLS.get(self.config.reid_model_name) or OFFICIAL_OSNET_URLS["osnet_x1_0_msmt17"]
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Downloading OSNet weights from %s -> %s...", url, dest_path)
            urllib.request.urlretrieve(url, str(dest_path))
            state_dict = torch.load(str(dest_path), map_location="cpu", weights_only=True)
            if isinstance(state_dict, dict) and "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            cleaned_state = {k.replace("module.", "").replace("backbone.", ""): v for k, v in state_dict.items()}
            self.model.load_state_dict(cleaned_state, strict=False)
            logger.info("Successfully downloaded and loaded OSNet weights.")
        except Exception as exc:
            logger.warning(
                "Offline or unable to auto-download weights (%s). Running with initialized weights. "
                "To install weights, place 'osnet_x1_0_msmt17.pth' in %s",
                exc,
                dest_path,
            )

    def embed_person_crop(
        self, person_crop_bgr: np.ndarray
    ) -> tuple[np.ndarray | None, Image.Image]:
        """Extracts a 512-D L2-normalized feature vector from a full-body person crop.

        Args:
            person_crop_bgr: Tight person bounding box crop from YOLO (BGR).

        Returns:
            (embedding_512_float32, person_crop_pil) or (None, pil_img) if crop is degenerate.
        """
        h, w = person_crop_bgr.shape[:2]
        if h < 20 or w < 10:
            return None, Image.fromarray(cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB))

        crop_rgb = cv2.cvtColor(person_crop_bgr, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)

        if self.model is None:
            return None, crop_pil

        tensor = REID_TRANSFORMS(crop_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            embedding = feat.cpu().numpy().flatten().astype(np.float32)

        return embedding, crop_pil
