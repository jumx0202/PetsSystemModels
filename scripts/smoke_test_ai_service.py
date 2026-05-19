"""Smoke test for the unified AI FastAPI service.

Usage:
    python scripts/smoke_test_ai_service.py --image path/to/pet.jpg
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


def post_image(url: str, field: str, image_path: Path) -> dict:
    with image_path.open("rb") as f:
        resp = requests.post(url, files={field: (image_path.name, f, "image/jpeg")}, timeout=120)
    resp.raise_for_status()
    return resp.json()


def post_verify(url: str, image_path: Path) -> dict:
    with image_path.open("rb") as f1, image_path.open("rb") as f2:
        resp = requests.post(
            url,
            files={
                "file1": (image_path.name, f1, "image/jpeg"),
                "file2": (image_path.name, f2, "image/jpeg"),
            },
            timeout=120,
        )
    resp.raise_for_status()
    return resp.json()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--image", required=True)
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    health = requests.get(f"{base_url}/health", timeout=30)
    health.raise_for_status()
    print("health:", health.json())

    breed = post_image(f"{base_url}/api/recognize", "file", image_path)
    print("breed:", breed)

    embedding = post_image(f"{base_url}/api/petface/embed", "file", image_path)
    data = embedding.get("data", {})
    print(
        "petface embed:",
        {
            "model_version": data.get("model_version"),
            "embedding_dim": data.get("embedding_dim"),
            "norm": data.get("norm"),
            "first5": data.get("embedding", [])[:5],
        },
    )

    verify = post_verify(f"{base_url}/api/petface/verify", image_path)
    print("petface verify same image:", verify)


if __name__ == "__main__":
    main()
