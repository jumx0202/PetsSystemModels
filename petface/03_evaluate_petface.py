"""
Evaluate PetFace-ID embeddings on official PetFace splits.

Typical quick run on the trained 9-animal model:
    python 03_evaluate_petface.py \
      --animals dog cat rabbit hamster guineapig chinchilla ferret parakeet hedgehog \
      --checkpoint models/petface_pet_species_full_continue/petface_id_best.pth \
      --tasks verification reid \
      --max-pairs 5000 \
      --max-reid-identities 3000 \
      --max-reid-queries 3000 \
      --gallery-per-id 3 \
      --batch-size 128 \
      --out models/petface_pet_species_full_continue/eval_quick.json

The re-identification task can become very large. Start with the quick command
above, then increase max-reid-identities/max-reid-queries if the result is
stable and runtime is acceptable.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, roc_auc_score
from tqdm import tqdm

from petface_dataset import PetFaceItem, read_labeled_split, read_verification_pairs
from petface_model import ModelConfig, build_transform, cosine_similarity, load_model, mean_embedding


IMAGE_LOAD_ERRORS = (OSError, ValueError)


def load_image(root: Path, filename: str) -> Image.Image:
    return Image.open(root / "images" / filename).convert("RGB")


@torch.no_grad()
def embed_filenames(
    root: Path,
    filenames: list[str],
    model,
    transform,
    device,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Embed unique filenames in batches and return a filename -> embedding map."""
    unique = list(dict.fromkeys(filenames))
    result: dict[str, np.ndarray] = {}
    for start in tqdm(range(0, len(unique), batch_size), desc="embed", unit="batch"):
        batch_files = unique[start:start + batch_size]
        images, valid_files = [], []
        for filename in batch_files:
            try:
                images.append(transform(load_image(root, filename)))
                valid_files.append(filename)
            except IMAGE_LOAD_ERRORS as exc:
                print(f"skip unreadable image: {filename} ({exc})")
        if not images:
            continue
        tensor = torch.stack(images).to(device)
        embeddings = model(tensor).detach().cpu().numpy().astype(np.float32)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / np.maximum(norms, 1e-12)
        for filename, embedding in zip(valid_files, embeddings):
            result[filename] = embedding
    return result


def evaluate_verification(
    root: Path,
    animal: str,
    model,
    transform,
    device,
    max_pairs: int | None,
    batch_size: int,
) -> dict:
    pair_path = root / "split" / animal / "verification.csv"
    pairs = read_verification_pairs(pair_path)
    if max_pairs:
        pairs = pairs[:max_pairs]

    filenames = []
    for f1, f2, _ in pairs:
        filenames.extend([f1, f2])
    cache = embed_filenames(root, filenames, model, transform, device, batch_size)

    scores, labels = [], []
    for f1, f2, label in tqdm(pairs, desc=f"verify {animal}", unit="pair"):
        if f1 not in cache or f2 not in cache:
            continue
        scores.append(cosine_similarity(cache[f1], cache[f2]))
        labels.append(label)

    scores_np = np.array(scores, dtype=np.float32)
    labels_np = np.array(labels, dtype=np.int64)
    auc = float(roc_auc_score(labels_np, scores_np)) if len(set(labels)) > 1 else 0.0
    best_acc, best_threshold = 0.0, 0.0
    for threshold in np.linspace(-0.2, 1.0, 241):
        preds = (scores_np >= threshold).astype(int)
        acc = float(accuracy_score(labels_np, preds))
        if acc > best_acc:
            best_acc = acc
            best_threshold = float(threshold)

    return {
        "animal": animal,
        "pairs": len(labels),
        "roc_auc": auc,
        "best_accuracy": best_acc,
        "best_threshold": best_threshold,
        "mean_positive_score": float(scores_np[labels_np == 1].mean()) if np.any(labels_np == 1) else None,
        "mean_negative_score": float(scores_np[labels_np == 0].mean()) if np.any(labels_np == 0) else None,
    }


def sample_items_for_reid(
    train_items: list[PetFaceItem],
    query_items: list[PetFaceItem],
    max_identities: int | None,
    max_queries: int | None,
    gallery_per_id: int,
    seed: int,
) -> tuple[list[PetFaceItem], list[PetFaceItem]]:
    rng = random.Random(seed)
    train_by_label: dict[int, list[PetFaceItem]] = defaultdict(list)
    query_by_label: dict[int, list[PetFaceItem]] = defaultdict(list)
    for item in train_items:
        train_by_label[item.label].append(item)
    for item in query_items:
        query_by_label[item.label].append(item)

    labels = sorted(set(train_by_label) & set(query_by_label))
    if max_identities and len(labels) > max_identities:
        labels = rng.sample(labels, max_identities)

    sampled_train, sampled_query = [], []
    for label in labels:
        gallery = train_by_label[label]
        queries = query_by_label[label]
        rng.shuffle(gallery)
        rng.shuffle(queries)
        sampled_train.extend(gallery[:gallery_per_id])
        sampled_query.extend(queries)

    rng.shuffle(sampled_query)
    if max_queries:
        sampled_query = sampled_query[:max_queries]
        query_labels = {item.label for item in sampled_query}
        sampled_train = [item for item in sampled_train if item.label in query_labels]

    return sampled_train, sampled_query


def evaluate_reidentification(
    root: Path,
    animal: str,
    model,
    transform,
    device,
    max_identities: int | None,
    max_queries: int | None,
    gallery_per_id: int,
    batch_size: int,
    sim_chunk_size: int,
    seed: int,
) -> dict:
    train_items = read_labeled_split(root / "split" / animal / "train.csv")
    query_items = read_labeled_split(root / "split" / animal / "reidentification.csv")
    train_items, query_items = sample_items_for_reid(
        train_items,
        query_items,
        max_identities=max_identities,
        max_queries=max_queries,
        gallery_per_id=gallery_per_id,
        seed=seed,
    )

    filenames = [item.filename for item in train_items] + [item.filename for item in query_items]
    cache = embed_filenames(root, filenames, model, transform, device, batch_size)

    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for item in train_items:
        if item.filename in cache:
            grouped[item.label].append(cache[item.filename])

    gallery_labels, gallery_embs = [], []
    for label, embs in grouped.items():
        if not embs:
            continue
        gallery_labels.append(label)
        gallery_embs.append(mean_embedding(embs))

    if not gallery_embs:
        return {"animal": animal, "queries": 0, "gallery_identities": 0, "top1": 0.0, "top5": 0.0}

    gallery = np.stack(gallery_embs).astype(np.float32)
    gallery_labels_np = np.array(gallery_labels)

    query_embs, query_labels = [], []
    for item in query_items:
        if item.filename in cache and item.label in grouped:
            query_embs.append(cache[item.filename])
            query_labels.append(item.label)

    top1, top5, total = 0, 0, 0
    for start in tqdm(range(0, len(query_embs), sim_chunk_size), desc=f"reid {animal}", unit="chunk"):
        query_batch = np.stack(query_embs[start:start + sim_chunk_size]).astype(np.float32)
        label_batch = np.array(query_labels[start:start + sim_chunk_size])
        sims = query_batch @ gallery.T
        topk = np.argsort(sims, axis=1)[:, -5:][:, ::-1]
        ranked_labels = gallery_labels_np[topk]
        total += len(label_batch)
        top1 += int((ranked_labels[:, 0] == label_batch).sum())
        top5 += int(sum(label in row for label, row in zip(label_batch, ranked_labels)))

    return {
        "animal": animal,
        "queries": total,
        "gallery_identities": len(gallery_labels),
        "gallery_images": len(train_items),
        "top1": top1 / max(total, 1),
        "top5": top5 / max(total, 1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("data/PetFace"))
    parser.add_argument("--animals", nargs="+", default=["dog", "cat"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--img-size", type=int, default=224)
    parser.add_argument("--embed-dim", type=int, default=512)
    parser.add_argument("--tasks", nargs="+", choices=["verification", "reid"], default=["verification", "reid"])
    parser.add_argument("--max-pairs", type=int, default=None)
    parser.add_argument("--max-reid-identities", type=int, default=3000)
    parser.add_argument("--max-reid-queries", type=int, default=3000)
    parser.add_argument("--gallery-per-id", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--sim-chunk-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=Path("models/eval_petface.json"))
    args = parser.parse_args()

    config = ModelConfig(
        model_name=args.model_name or "hf-hub:BVRA/MegaDescriptor-B-224",
        img_size=args.img_size,
        embed_dim=args.embed_dim,
        checkpoint=args.checkpoint,
    )
    model, device = load_model(config)
    transform = build_transform(args.img_size)
    report = {
        "model_name": config.model_name,
        "checkpoint": args.checkpoint,
        "img_size": args.img_size,
        "embed_dim": args.embed_dim,
        "animals": args.animals,
        "tasks": args.tasks,
        "verification": [],
        "reidentification": [],
    }

    for animal in args.animals:
        if "verification" in args.tasks and (args.root / "split" / animal / "verification.csv").exists():
            report["verification"].append(evaluate_verification(
                args.root, animal, model, transform, device, args.max_pairs, args.batch_size
            ))
        if "reid" in args.tasks and (args.root / "split" / animal / "reidentification.csv").exists():
            report["reidentification"].append(evaluate_reidentification(
                args.root,
                animal,
                model,
                transform,
                device,
                args.max_reid_identities,
                args.max_reid_queries,
                args.gallery_per_id,
                args.batch_size,
                args.sim_chunk_size,
                args.seed,
            ))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
