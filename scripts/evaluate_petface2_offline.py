#!/usr/bin/env python3
"""
Offline PetFace-ID 2.0 evaluator.

This script validates the same-pet recognition capability without depending on
Spring Boot, MySQL, or the frontend. It loads the local PetFace 2.0 checkpoint
directly and evaluates it against local PetFace images.

Supported evaluation tasks:
  1. verification:
     Use official verification.csv image pairs and measure whether two images
     belong to the same individual.

  2. reid:
     Use official train.csv as gallery and reidentification.csv as query, then
     report Top-1 / Top-5 retrieval accuracy.

  3. closed_loop:
     Simulate the system workflow directly from images:
       registered profile images -> gallery
       later uploaded images      -> query
     This is the closest offline approximation of:
       pet profile -> visual feature -> lost-pet photo search.

Typical quick run after the current MPS training is idle:
    cd ai_recognition
    conda activate pet
    export KMP_DUPLICATE_LIB_OK=TRUE

    python scripts/evaluate_petface2_offline.py \\
      --petface-root "/Volumes/ORGOS - Data/PetFaceWorkspace/ai_petface/data/PetFace" \\
      --animals dog cat rabbit hamster guineapig chinchilla ferret parakeet hedgehog \\
      --tasks verification reid closed_loop \\
      --checkpoint models/petface/petface_id_best.pth \\
      --batch-size 16 \\
      --max-pairs 3000 \\
      --max-reid-identities 1000 \\
      --max-reid-queries 1000 \\
      --max-closed-loop-identities 1000 \\
      --out reports/petface2_eval_quick.json

For a more complete but slower run, increase or remove the max-* limits.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
AI_DIR = SCRIPT_DIR.parent
if str(AI_DIR) not in sys.path:
    sys.path.insert(0, str(AI_DIR))

from petface.petface_model import (  # noqa: E402
    ModelConfig,
    build_transform,
    cosine_similarity,
    load_model,
    mean_embedding,
)


DEFAULT_PETFACE_ROOT = Path(
    "/Volumes/ORGOS - Data/PetFaceWorkspace/ai_petface/data/PetFace"
)
DEFAULT_CHECKPOINT = AI_DIR / "models" / "petface" / "petface_id_best.pth"
DEFAULT_OUT = AI_DIR / "reports" / "petface2_eval_offline.json"
DEFAULT_ANIMALS = [
    "dog",
    "cat",
    "rabbit",
    "hamster",
    "guineapig",
    "chinchilla",
    "ferret",
    "parakeet",
    "hedgehog",
]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
IMAGE_ERRORS = (OSError, ValueError, UnidentifiedImageError)


@dataclass(frozen=True)
class LabeledImage:
    filename: str
    label: str


@dataclass(frozen=True)
class VerificationPair:
    filename1: str
    filename2: str
    label: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate PetFace-ID 2.0 offline without Spring Boot."
    )
    parser.add_argument("--petface-root", type=Path, default=DEFAULT_PETFACE_ROOT)
    parser.add_argument("--animals", nargs="+", default=DEFAULT_ANIMALS)
    parser.add_argument(
        "--tasks",
        nargs="+",
        choices=["verification", "reid", "closed_loop"],
        default=["verification", "reid", "closed_loop"],
    )
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--model-name", default="hf-hub:BVRA/MegaDescriptor-B-224")
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "mps", "cuda"],
        default="auto",
        help="Inference device. Use cpu for the most stable local evaluation.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--sim-chunk-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=20260519)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    parser.add_argument("--max-pairs", type=int, default=3000)
    parser.add_argument("--max-reid-identities", type=int, default=1000)
    parser.add_argument("--max-reid-queries", type=int, default=1000)
    parser.add_argument("--gallery-per-id", type=int, default=3)

    parser.add_argument("--max-closed-loop-identities", type=int, default=1000)
    parser.add_argument("--closed-loop-gallery-per-id", type=int, default=1)
    parser.add_argument("--closed-loop-query-per-id", type=int, default=1)
    parser.add_argument(
        "--save-scores",
        action="store_true",
        help="Include per-pair / per-query scores in JSON. This can make reports large.",
    )
    return parser.parse_args()


def log(message: str) -> None:
    print(f"[petface-eval] {message}", flush=True)


def resolve_device(name: str) -> torch.device | None:
    if name == "auto":
        return None
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS is not available in this Python environment.")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this Python environment.")
    return torch.device(name)


def identity_from_filename(filename: str) -> str:
    parts = Path(filename).parts
    if len(parts) >= 2:
        return f"{parts[0]}/{parts[1]}"
    return filename


def image_path(root: Path, filename: str) -> Path:
    return root / "images" / filename


def read_labeled_split(path: Path) -> list[LabeledImage]:
    if not path.exists():
        return []

    items: list[LabeledImage] = []
    if path.suffix.lower() == ".txt":
        with path.open(encoding="utf-8") as f:
            for line in f:
                filename = line.strip()
                if filename:
                    items.append(LabeledImage(filename, identity_from_filename(filename)))
        return items

    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            filename = row.get("filename") or row.get("file") or row.get("path")
            if not filename:
                continue
            label = row.get("label") or identity_from_filename(filename)
            items.append(LabeledImage(filename, str(label)))
    return items


def read_verification_pairs(path: Path) -> list[VerificationPair]:
    if not path.exists():
        return []
    pairs: list[VerificationPair] = []
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            f1 = row.get("filename1") or row.get("file1") or row.get("path1")
            f2 = row.get("filename2") or row.get("file2") or row.get("path2")
            label = row.get("label") or row.get("same")
            if f1 and f2 and label is not None:
                pairs.append(VerificationPair(f1, f2, int(label)))
    return pairs


def list_identity_images(root: Path, animal: str) -> dict[str, list[str]]:
    animal_dir = root / "images" / animal
    identities: dict[str, list[str]] = {}
    if not animal_dir.exists():
        return identities
    log(f"scanning {animal_dir}")
    for identity_dir in sorted(p for p in animal_dir.iterdir() if p.is_dir()):
        files = [
            f"{animal}/{identity_dir.name}/{p.name}"
            for p in sorted(identity_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if len(files) >= 2:
            identities[f"{animal}/{identity_dir.name}"] = files
    return identities


def list_identity_images_from_split(root: Path, animal: str) -> dict[str, list[str]]:
    """Build identity groups from train.csv without scanning huge image dirs."""
    split_path = root / "split" / animal / "train.csv"
    items = read_labeled_split(split_path)
    identities: dict[str, list[str]] = defaultdict(list)
    for item in items:
        identities[f"{animal}/{item.label}"].append(item.filename)
    return {
        label: files
        for label, files in identities.items()
        if len(files) >= 2
    }


def take_sample(values: list[Any], limit: int | None, rng: random.Random) -> list[Any]:
    if limit is None or limit <= 0 or len(values) <= limit:
        return values
    return rng.sample(values, limit)


def safe_float(value: float | np.floating | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def percentile(values: np.ndarray, q: float) -> float | None:
    if values.size == 0:
        return None
    return safe_float(np.percentile(values, q))


def binary_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = scores >= threshold
    positives = labels == 1
    negatives = labels == 0
    tp = int(np.logical_and(preds, positives).sum())
    tn = int(np.logical_and(~preds, negatives).sum())
    fp = int(np.logical_and(preds, negatives).sum())
    fn = int(np.logical_and(~preds, positives).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    accuracy = (tp + tn) / max(len(labels), 1)
    return {
        "threshold": threshold,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float | None:
    positives = scores[labels == 1]
    negatives = scores[labels == 0]
    if positives.size == 0 or negatives.size == 0:
        return None

    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    sorted_scores = scores[order]
    start = 0
    while start < len(sorted_scores):
        end = start + 1
        while end < len(sorted_scores) and sorted_scores[end] == sorted_scores[start]:
            end += 1
        avg_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = avg_rank
        start = end

    pos_ranks = ranks[labels == 1].sum()
    n_pos = positives.size
    n_neg = negatives.size
    return safe_float((pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


@torch.no_grad()
def embed_filenames(
    root: Path,
    filenames: Iterable[str],
    model,
    transform,
    device,
    batch_size: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, str]]]:
    unique = list(dict.fromkeys(filenames))
    embeddings: dict[str, np.ndarray] = {}
    errors: list[dict[str, str]] = []
    log(f"embedding {len(unique)} unique images on {device} with batch_size={batch_size}")

    for start in tqdm(range(0, len(unique), batch_size), desc="embedding", unit="batch"):
        batch_names = unique[start:start + batch_size]
        tensors = []
        valid_names = []
        for filename in batch_names:
            path = image_path(root, filename)
            try:
                image = Image.open(path).convert("RGB")
                tensors.append(transform(image))
                valid_names.append(filename)
            except IMAGE_ERRORS as exc:
                errors.append({"filename": filename, "error": str(exc)})
        if not tensors:
            continue

        batch = torch.stack(tensors).to(device)
        emb = model(batch).detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        emb = emb / np.maximum(norms, 1e-12)
        for filename, vector in zip(valid_names, emb):
            embeddings[filename] = vector

    return embeddings, errors


def summarize_scores(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    return {
        "pairs": int(len(labels)),
        "positive_pairs": int(pos.size),
        "negative_pairs": int(neg.size),
        "roc_auc": roc_auc(labels, scores),
        "mean_positive_score": safe_float(pos.mean()) if pos.size else None,
        "mean_negative_score": safe_float(neg.mean()) if neg.size else None,
        "p05_positive_score": percentile(pos, 5),
        "p50_positive_score": percentile(pos, 50),
        "p95_negative_score": percentile(neg, 95),
    }


def best_threshold_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    if labels.size == 0:
        return {}
    candidates = np.linspace(-0.2, 1.0, 241)
    best = None
    for threshold in candidates:
        metrics = binary_metrics(labels, scores, float(threshold))
        if best is None or metrics["accuracy"] > best["accuracy"]:
            best = metrics
    return best or {}


def evaluate_verification(
    root: Path,
    animal: str,
    model,
    transform,
    device,
    batch_size: int,
    max_pairs: int | None,
    threshold: float,
    rng: random.Random,
    save_scores: bool,
) -> dict[str, Any]:
    pairs = read_verification_pairs(root / "split" / animal / "verification.csv")
    pairs = take_sample(pairs, max_pairs, rng)
    filenames = []
    for pair in pairs:
        filenames.extend([pair.filename1, pair.filename2])

    cache, errors = embed_filenames(root, filenames, model, transform, device, batch_size)
    labels, scores, details = [], [], []
    for pair in tqdm(pairs, desc=f"verification {animal}", unit="pair"):
        emb1 = cache.get(pair.filename1)
        emb2 = cache.get(pair.filename2)
        if emb1 is None or emb2 is None:
            continue
        score = cosine_similarity(emb1, emb2)
        labels.append(pair.label)
        scores.append(score)
        if save_scores:
            details.append({
                "filename1": pair.filename1,
                "filename2": pair.filename2,
                "label": pair.label,
                "score": score,
            })

    labels_np = np.array(labels, dtype=np.int64)
    scores_np = np.array(scores, dtype=np.float32)
    result = {
        "animal": animal,
        "source": "official verification.csv",
        "requested_pairs": len(pairs),
        "evaluated_pairs": int(len(labels_np)),
        "image_errors": errors[:20],
        "image_error_count": len(errors),
        "score_summary": summarize_scores(labels_np, scores_np) if len(labels_np) else {},
        "at_config_threshold": binary_metrics(labels_np, scores_np, threshold) if len(labels_np) else {},
        "best_threshold": best_threshold_metrics(labels_np, scores_np),
    }
    if save_scores:
        result["scores"] = details
    return result


def sample_reid_items(
    train_items: list[LabeledImage],
    query_items: list[LabeledImage],
    max_identities: int | None,
    max_queries: int | None,
    gallery_per_id: int,
    rng: random.Random,
) -> tuple[list[LabeledImage], list[LabeledImage]]:
    train_by_id: dict[str, list[LabeledImage]] = defaultdict(list)
    query_by_id: dict[str, list[LabeledImage]] = defaultdict(list)
    for item in train_items:
        train_by_id[item.label].append(item)
    for item in query_items:
        query_by_id[item.label].append(item)

    labels = sorted(set(train_by_id) & set(query_by_id))
    labels = take_sample(labels, max_identities, rng)

    gallery, queries = [], []
    for label in labels:
        g = train_by_id[label][:]
        q = query_by_id[label][:]
        rng.shuffle(g)
        rng.shuffle(q)
        gallery.extend(g[:gallery_per_id])
        queries.extend(q)

    rng.shuffle(queries)
    queries = take_sample(queries, max_queries, rng)
    query_labels = {item.label for item in queries}
    gallery = [item for item in gallery if item.label in query_labels]
    return gallery, queries


def evaluate_reid_like(
    *,
    root: Path,
    animal: str,
    gallery_items: list[LabeledImage],
    query_items: list[LabeledImage],
    source: str,
    model,
    transform,
    device,
    batch_size: int,
    sim_chunk_size: int,
    save_scores: bool,
) -> dict[str, Any]:
    filenames = [item.filename for item in gallery_items] + [item.filename for item in query_items]
    cache, errors = embed_filenames(root, filenames, model, transform, device, batch_size)

    grouped: dict[str, list[np.ndarray]] = defaultdict(list)
    for item in gallery_items:
        if item.filename in cache:
            grouped[item.label].append(cache[item.filename])

    gallery_labels, gallery_embs = [], []
    for label, embs in grouped.items():
        if embs:
            gallery_labels.append(label)
            gallery_embs.append(mean_embedding(embs))

    if not gallery_embs:
        return {
            "animal": animal,
            "source": source,
            "queries": 0,
            "gallery_identities": 0,
            "top1": 0.0,
            "top5": 0.0,
            "image_error_count": len(errors),
            "image_errors": errors[:20],
        }

    gallery_matrix = np.stack(gallery_embs).astype(np.float32)
    gallery_labels_np = np.array(gallery_labels)

    valid_queries: list[tuple[LabeledImage, np.ndarray]] = []
    for item in query_items:
        emb = cache.get(item.filename)
        if emb is not None and item.label in grouped:
            valid_queries.append((item, emb))

    top1 = 0
    top5 = 0
    reciprocal_rank_sum = 0.0
    details = []
    top1_scores = []
    correct_scores = []

    for start in tqdm(
        range(0, len(valid_queries), sim_chunk_size),
        desc=f"{source} {animal}",
        unit="chunk",
    ):
        batch = valid_queries[start:start + sim_chunk_size]
        query_matrix = np.stack([emb for _, emb in batch]).astype(np.float32)
        true_labels = np.array([item.label for item, _ in batch])
        sims = query_matrix @ gallery_matrix.T
        order = np.argsort(sims, axis=1)[:, ::-1]
        ranked_labels = gallery_labels_np[order[:, :5]]
        ranked_scores = np.take_along_axis(sims, order[:, :5], axis=1)

        for row_idx, (item, _) in enumerate(batch):
            row_labels = ranked_labels[row_idx]
            row_scores = ranked_scores[row_idx]
            match_positions = np.where(gallery_labels_np[order[row_idx]] == item.label)[0]
            rank = int(match_positions[0] + 1) if match_positions.size else None
            is_top1 = row_labels[0] == item.label
            is_top5 = item.label in set(row_labels.tolist())
            top1 += int(is_top1)
            top5 += int(is_top5)
            reciprocal_rank_sum += 0.0 if rank is None else 1.0 / rank
            top1_scores.append(float(row_scores[0]))
            if rank is not None:
                correct_scores.append(float(sims[row_idx, order[row_idx, rank - 1]]))
            if save_scores:
                details.append({
                    "query": item.filename,
                    "label": item.label,
                    "rank": rank,
                    "top5_labels": row_labels.tolist(),
                    "top5_scores": [float(v) for v in row_scores],
                })

    total = len(valid_queries)
    result = {
        "animal": animal,
        "source": source,
        "queries": total,
        "gallery_identities": len(gallery_labels),
        "gallery_images": len(gallery_items),
        "top1": top1 / max(total, 1),
        "top5": top5 / max(total, 1),
        "mean_reciprocal_rank": reciprocal_rank_sum / max(total, 1),
        "mean_top1_score": safe_float(np.mean(top1_scores)) if top1_scores else None,
        "mean_correct_identity_score": safe_float(np.mean(correct_scores)) if correct_scores else None,
        "image_error_count": len(errors),
        "image_errors": errors[:20],
    }
    if save_scores:
        result["queries_detail"] = details
    return result


def evaluate_reidentification(
    root: Path,
    animal: str,
    model,
    transform,
    device,
    batch_size: int,
    sim_chunk_size: int,
    max_identities: int | None,
    max_queries: int | None,
    gallery_per_id: int,
    rng: random.Random,
    save_scores: bool,
) -> dict[str, Any]:
    train_items = read_labeled_split(root / "split" / animal / "train.csv")
    query_items = read_labeled_split(root / "split" / animal / "reidentification.csv")
    gallery, queries = sample_reid_items(
        train_items,
        query_items,
        max_identities=max_identities,
        max_queries=max_queries,
        gallery_per_id=gallery_per_id,
        rng=rng,
    )
    return evaluate_reid_like(
        root=root,
        animal=animal,
        gallery_items=gallery,
        query_items=queries,
        source="official reidentification",
        model=model,
        transform=transform,
        device=device,
        batch_size=batch_size,
        sim_chunk_size=sim_chunk_size,
        save_scores=save_scores,
    )


def evaluate_closed_loop(
    root: Path,
    animal: str,
    model,
    transform,
    device,
    batch_size: int,
    sim_chunk_size: int,
    max_identities: int | None,
    gallery_per_id: int,
    query_per_id: int,
    rng: random.Random,
    save_scores: bool,
) -> dict[str, Any]:
    identities = list_identity_images_from_split(root, animal)
    source = "closed_loop profile-to-query simulation from train.csv"
    log(f"{animal}: loaded {len(identities)} closed_loop identities from split/train.csv")
    if not identities:
        identities = list_identity_images(root, animal)
        source = "closed_loop profile-to-query simulation from image directories"
        log(f"{animal}: loaded {len(identities)} closed_loop identities from image directories")
    identity_keys = take_sample(sorted(identities), max_identities, rng)
    log(f"{animal}: selected {len(identity_keys)} identities for closed_loop")
    gallery, queries = [], []
    for label in identity_keys:
        files = identities[label][:]
        rng.shuffle(files)
        if len(files) <= gallery_per_id:
            continue
        gallery_files = files[:gallery_per_id]
        query_files = files[gallery_per_id:gallery_per_id + query_per_id]
        gallery.extend(LabeledImage(filename=f, label=label) for f in gallery_files)
        queries.extend(LabeledImage(filename=f, label=label) for f in query_files)

    return evaluate_reid_like(
        root=root,
        animal=animal,
        gallery_items=gallery,
        query_items=queries,
        source=source,
        model=model,
        transform=transform,
        device=device,
        batch_size=batch_size,
        sim_chunk_size=sim_chunk_size,
        save_scores=save_scores,
    )


def aggregate_reid(results: list[dict[str, Any]], key: str) -> dict[str, Any]:
    total_queries = sum(int(r.get("queries", 0)) for r in results)
    if total_queries == 0:
        return {}
    return {
        "queries": total_queries,
        "weighted_top1": sum(r.get("top1", 0.0) * r.get("queries", 0) for r in results) / total_queries,
        "weighted_top5": sum(r.get("top5", 0.0) * r.get("queries", 0) for r in results) / total_queries,
        "weighted_mrr": sum(r.get("mean_reciprocal_rank", 0.0) * r.get("queries", 0) for r in results) / total_queries,
        "metric": key,
    }


def main() -> None:
    args = parse_args()
    rng = random.Random(args.seed)

    log(f"root={args.petface_root}")
    log(f"checkpoint={args.checkpoint}")
    log(f"tasks={','.join(args.tasks)} animals={','.join(args.animals)}")

    if not args.petface_root.exists():
        raise FileNotFoundError(f"PetFace root not found: {args.petface_root}")
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"PetFace checkpoint not found: {args.checkpoint}")

    config = ModelConfig(
        model_name=args.model_name,
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        checkpoint=str(args.checkpoint),
    )
    log(f"loading model: {args.model_name}")
    model, device = load_model(config, device=resolve_device(args.device))
    log(f"model loaded on device={device}")
    transform = build_transform(args.img_size)

    report: dict[str, Any] = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "petface_root": str(args.petface_root),
        "checkpoint": str(args.checkpoint),
        "model_name": args.model_name,
        "img_size": args.img_size,
        "embed_dim": args.embed_dim,
        "device": str(device),
        "animals": args.animals,
        "tasks": args.tasks,
        "threshold": args.threshold,
        "limits": {
            "max_pairs": args.max_pairs,
            "max_reid_identities": args.max_reid_identities,
            "max_reid_queries": args.max_reid_queries,
            "gallery_per_id": args.gallery_per_id,
            "max_closed_loop_identities": args.max_closed_loop_identities,
            "closed_loop_gallery_per_id": args.closed_loop_gallery_per_id,
            "closed_loop_query_per_id": args.closed_loop_query_per_id,
        },
        "verification": [],
        "reidentification": [],
        "closed_loop": [],
    }

    for animal in args.animals:
        log(f"start animal={animal}")
        split_dir = args.petface_root / "split" / animal
        image_dir = args.petface_root / "images" / animal
        if not split_dir.exists() and not image_dir.exists():
            print(f"[warn] skip missing animal: {animal}", file=sys.stderr)
            continue

        if "verification" in args.tasks:
            pair_path = split_dir / "verification.csv"
            if pair_path.exists():
                report["verification"].append(evaluate_verification(
                    args.petface_root,
                    animal,
                    model,
                    transform,
                    device,
                    args.batch_size,
                    args.max_pairs,
                    args.threshold,
                    rng,
                    args.save_scores,
                ))
            else:
                print(f"[warn] skip verification without file: {pair_path}", file=sys.stderr)

        if "reid" in args.tasks:
            train_path = split_dir / "train.csv"
            query_path = split_dir / "reidentification.csv"
            if train_path.exists() and query_path.exists():
                report["reidentification"].append(evaluate_reidentification(
                    args.petface_root,
                    animal,
                    model,
                    transform,
                    device,
                    args.batch_size,
                    args.sim_chunk_size,
                    args.max_reid_identities,
                    args.max_reid_queries,
                    args.gallery_per_id,
                    rng,
                    args.save_scores,
                ))
            else:
                print(f"[warn] skip reid without files: {train_path}, {query_path}", file=sys.stderr)

        if "closed_loop" in args.tasks:
            if image_dir.exists():
                report["closed_loop"].append(evaluate_closed_loop(
                    args.petface_root,
                    animal,
                    model,
                    transform,
                    device,
                    args.batch_size,
                    args.sim_chunk_size,
                    args.max_closed_loop_identities,
                    args.closed_loop_gallery_per_id,
                    args.closed_loop_query_per_id,
                    rng,
                    args.save_scores,
                ))
            else:
                print(f"[warn] skip closed_loop without images: {image_dir}", file=sys.stderr)

    report["summary"] = {
        "reidentification": aggregate_reid(report["reidentification"], "top1/top5/mrr"),
        "closed_loop": aggregate_reid(report["closed_loop"], "top1/top5/mrr"),
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
