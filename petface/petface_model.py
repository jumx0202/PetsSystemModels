"""
PetFace-ID 2.0 model utilities.

The default backbone uses MegaDescriptor from Hugging Face through timm:
    hf-hub:BVRA/MegaDescriptor-B-224

MegaDescriptor is a strong animal re-identification feature extractor and is
therefore a practical high-quality default while PetFace dataset access and
fine-tuning are being prepared.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
import timm


DEFAULT_MODEL_NAME = "hf-hub:BVRA/MegaDescriptor-B-224"
DEFAULT_IMG_SIZE = 224
DEFAULT_EMBED_DIM = 512


@dataclass
class ModelConfig:
    model_name: str = DEFAULT_MODEL_NAME
    img_size: int = DEFAULT_IMG_SIZE
    embed_dim: int = DEFAULT_EMBED_DIM
    checkpoint: str | None = None


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def build_transform(img_size: int = DEFAULT_IMG_SIZE) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class EmbeddingNet(nn.Module):
    """Backbone + projection head, returning L2-normalized embeddings."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        embed_dim: int = DEFAULT_EMBED_DIM,
        pretrained: bool = True,
        use_projector: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
        feature_dim = self.backbone.num_features
        self.output_dim = embed_dim if use_projector else feature_dim
        self.projector = (
            nn.Sequential(
                nn.Linear(feature_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
            )
            if use_projector
            else nn.Identity()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        embeddings = self.projector(features)
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)


def load_model(config: ModelConfig | None = None, device: torch.device | None = None) -> tuple[EmbeddingNet, torch.device]:
    config = config or ModelConfig()
    device = device or get_device()
    model = EmbeddingNet(
        model_name=config.model_name,
        embed_dim=config.embed_dim,
        pretrained=config.checkpoint is None,
        use_projector=config.checkpoint is not None,
    )
    if config.checkpoint:
        state = torch.load(config.checkpoint, map_location="cpu")
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def image_to_tensor(image: Image.Image, transform: transforms.Compose) -> torch.Tensor:
    return transform(image.convert("RGB")).unsqueeze(0)


@torch.no_grad()
def embed_image(
    model: nn.Module,
    image: Image.Image,
    transform: transforms.Compose,
    device: torch.device,
) -> np.ndarray:
    tensor = image_to_tensor(image, transform).to(device)
    embedding = model(tensor).detach().cpu().numpy()[0].astype(np.float32)
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm
    return embedding


@torch.no_grad()
def embed_paths(
    model: nn.Module,
    image_paths: Iterable[Path],
    transform: transforms.Compose,
    device: torch.device,
) -> list[np.ndarray]:
    embeddings: list[np.ndarray] = []
    for path in image_paths:
        image = Image.open(path).convert("RGB")
        embeddings.append(embed_image(model, image, transform, device))
    return embeddings


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32)
    b = b.astype(np.float32)
    a_norm = np.linalg.norm(a)
    b_norm = np.linalg.norm(b)
    if a_norm == 0 or b_norm == 0:
        return 0.0
    return float(np.dot(a / a_norm, b / b_norm))


def mean_embedding(embeddings: list[np.ndarray]) -> np.ndarray:
    if not embeddings:
        raise ValueError("empty embedding list")
    avg = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm = np.linalg.norm(avg)
    return avg / norm if norm > 0 else avg
