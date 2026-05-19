"""
构建扩展版宠物品种识别数据集。

输入：
  1. 原 Oxford-IIIT Pet 已整理数据：dataset/train|val|test
  2. Stanford Dogs 原始目录：data_raw/stanford_dogs/Images
  3. Kaggle CatBreedsRefined 原始目录：data_raw/kaggle_catbreedsrefined/CatBreedsRefined-v2

输出：
  dataset_extended/train|val|test/<breed>/*.jpg

运行：
  python 07_build_extended_dataset.py
"""

import argparse
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DEFAULT_OXFORD_DIR = BASE_DIR / "dataset"
DEFAULT_STANFORD_DIR = BASE_DIR / "data_raw" / "stanford_dogs" / "Images"
DEFAULT_KAGGLE_CATS_DIR = BASE_DIR / "data_raw" / "kaggle_catbreedsrefined" / "CatBreedsRefined-v2"
DEFAULT_OUT_DIR = BASE_DIR / "dataset_extended"
DEFAULT_META_FILE = BASE_DIR / "models_extended" / "class_meta.json"

SPLIT_RATIO = (0.8, 0.1, 0.1)
RANDOM_SEED = 42

CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll",
    "Russian_Blue", "Siamese", "Sphynx",
    "American_Bobtail", "American_Curl", "American_Shorthair",
    "Exotic_Shorthair", "Manx", "Norwegian_Forest",
    "Scottish_Fold", "Turkish_Angora",
}

# Stanford Dogs 与 Oxford-IIIT Pet 的少量同义类别统一到 Oxford 原名。
BREED_ALIASES = {
    "basset": "basset_hound",
    "cocker_spaniel": "english_cocker_spaniel",
    "german_short_haired_pointer": "german_shorthaired",
    "japanese_spaniel": "japanese_chin",
    "leonberg": "leonberger",
    "saint_bernard": "saint_bernard",
    "great_pyrenees": "great_pyrenees",
    "scotch_terrier": "scottish_terrier",
    "soft_coated_wheaten_terrier": "wheaten_terrier",
    "american_staffordshire_terrier": "american_pit_bull_terrier",
    "staffordshire_bullterrier": "staffordshire_bull_terrier",
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
NON_PET_CANIDS = {"african_hunting_dog", "dhole", "dingo"}

# Kaggle CatBreedsRefined 与 Oxford-IIIT Pet 的猫品种统一命名。
CAT_BREED_ALIASES = {
    "abyssinian": "Abyssinian",
    "bengal": "Bengal",
    "birman": "Birman",
    "bombay": "Bombay",
    "british_shorthair": "British_Shorthair",
    "egyptian_mau": "Egyptian_Mau",
    "maine_coon": "Maine_Coon",
    "persian": "Persian",
    "ragdoll": "Ragdoll",
    "russian_blue": "Russian_Blue",
    "siamese": "Siamese",
    "sphynx": "Sphynx",
    "american_bobtail": "American_Bobtail",
    "american_curl": "American_Curl",
    "american_shorthair": "American_Shorthair",
    "exotic_shorthair": "Exotic_Shorthair",
    "manx": "Manx",
    "norwegian_forest": "Norwegian_Forest",
    "scottish_fold": "Scottish_Fold",
    "turkish_angora": "Turkish_Angora",
}


def normalize_breed(raw_name: str) -> str:
    name = raw_name.strip()
    if "-" in name and re.match(r"^n\d+-", name):
        name = name.split("-", 1)[1]
    name = name.replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return BREED_ALIASES.get(name, name)


def normalize_cat_breed(raw_name: str) -> str:
    name = raw_name.strip().replace("-", "_").replace(" ", "_")
    name = re.sub(r"[^0-9A-Za-z_]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("_").lower()
    return CAT_BREED_ALIASES.get(name, "_".join(part.capitalize() for part in name.split("_")))


def link_or_copy(src: Path, dest: Path, use_hardlink: bool):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    if use_hardlink:
        try:
            dest.hardlink_to(src)
            return
        except OSError:
            pass
    shutil.copy2(src, dest)


def collect_oxford(oxford_dir: Path, out_dir: Path, use_hardlink: bool) -> dict[str, int]:
    stats = defaultdict(int)
    for split in ("train", "val", "test"):
        split_dir = oxford_dir / split
        if not split_dir.exists():
            raise FileNotFoundError(f"未找到 Oxford 已整理目录：{split_dir}")

        for cls_dir in tqdm(
            sorted(p for p in split_dir.iterdir() if p.is_dir()),
            desc=f"复制 Oxford {split}",
            unit="类",
        ):
            cls_name = cls_dir.name
            for src in cls_dir.iterdir():
                if src.is_file() and src.suffix.lower() in IMAGE_EXTS:
                    dest = out_dir / split / cls_name / src.name
                    link_or_copy(src, dest, use_hardlink)
                    stats[cls_name] += 1
    return dict(stats)


def collect_stanford(stanford_dir: Path, include_non_pet: bool) -> dict[str, list[Path]]:
    if not stanford_dir.exists():
        raise FileNotFoundError(
            f"未找到 Stanford Dogs 图片目录：{stanford_dir}\n"
            "请先运行 python 06_download_stanford_dogs.py"
        )

    grouped: dict[str, list[Path]] = defaultdict(list)
    for cls_dir in sorted(p for p in stanford_dir.iterdir() if p.is_dir()):
        breed = normalize_breed(cls_dir.name)
        if not include_non_pet and breed in NON_PET_CANIDS:
            continue
        for img in cls_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                grouped[breed].append(img)
    return grouped


def add_stanford(
    grouped: dict[str, list[Path]],
    out_dir: Path,
    use_hardlink: bool,
) -> dict[str, int]:
    stats = defaultdict(int)
    rng = random.Random(RANDOM_SEED)

    for breed, imgs in tqdm(sorted(grouped.items()), desc="合并 Stanford Dogs", unit="类"):
        imgs = list(imgs)
        rng.shuffle(imgs)
        n = len(imgs)
        t_end = int(n * SPLIT_RATIO[0])
        v_end = t_end + int(n * SPLIT_RATIO[1])
        split_map = {
            "train": imgs[:t_end],
            "val": imgs[t_end:v_end],
            "test": imgs[v_end:],
        }

        for split, files in split_map.items():
            for src in files:
                safe_parent = normalize_breed(src.parent.name)
                dest_name = f"stanford_{safe_parent}_{src.name}"
                dest = out_dir / split / breed / dest_name
                link_or_copy(src, dest, use_hardlink)
                stats[breed] += 1
    return dict(stats)


def collect_kaggle_cats(kaggle_cats_dir: Path) -> dict[str, list[Path]]:
    if not kaggle_cats_dir.exists():
        print(f"跳过 Kaggle 猫品种数据集，目录不存在：{kaggle_cats_dir}")
        return {}

    grouped: dict[str, list[Path]] = defaultdict(list)
    for cls_dir in sorted(p for p in kaggle_cats_dir.iterdir() if p.is_dir()):
        breed = normalize_cat_breed(cls_dir.name)
        for img in cls_dir.iterdir():
            if img.is_file() and img.suffix.lower() in IMAGE_EXTS:
                grouped[breed].append(img)
    return grouped


def add_kaggle_cats(
    grouped: dict[str, list[Path]],
    out_dir: Path,
    use_hardlink: bool,
) -> dict[str, int]:
    stats = defaultdict(int)
    rng = random.Random(RANDOM_SEED)

    for breed, imgs in tqdm(sorted(grouped.items()), desc="合并 Kaggle Cats", unit="类"):
        imgs = list(imgs)
        rng.shuffle(imgs)
        n = len(imgs)
        t_end = int(n * SPLIT_RATIO[0])
        v_end = t_end + int(n * SPLIT_RATIO[1])
        split_map = {
            "train": imgs[:t_end],
            "val": imgs[t_end:v_end],
            "test": imgs[v_end:],
        }

        for split, files in split_map.items():
            for src in files:
                safe_parent = normalize_cat_breed(src.parent.name)
                dest_name = f"kaggle_catbreedsrefined_{safe_parent}_{src.name}"
                dest = out_dir / split / breed / dest_name
                link_or_copy(src, dest, use_hardlink)
                stats[breed] += 1
    return dict(stats)


def build_meta(out_dir: Path, meta_file: Path):
    train_dir = out_dir / "train"
    classes = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {str(i): cls for cls, i in class_to_idx.items()}
    class_type = {cls: ("cat" if cls in CAT_BREEDS else "dog") for cls in classes}

    meta_file.parent.mkdir(parents=True, exist_ok=True)
    with open(meta_file, "w", encoding="utf-8") as f:
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": idx_to_class,
            "class_type": class_type,
            "num_classes": len(classes),
            "data_dir": str(out_dir),
            "source": "Oxford-IIIT Pet + Stanford Dogs + Kaggle CatBreedsRefined-7k",
        }, f, ensure_ascii=False, indent=2)

    return classes, class_type


def count_images(out_dir: Path) -> dict[str, int]:
    counts = {}
    for split in ("train", "val", "test"):
        counts[split] = sum(
            1
            for p in (out_dir / split).rglob("*")
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        )
    return counts


def main():
    parser = argparse.ArgumentParser(description="构建扩展版宠物品种识别数据集")
    parser.add_argument("--oxford-dir", type=Path, default=DEFAULT_OXFORD_DIR)
    parser.add_argument("--stanford-dir", type=Path, default=DEFAULT_STANFORD_DIR)
    parser.add_argument("--kaggle-cats-dir", type=Path, default=DEFAULT_KAGGLE_CATS_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--meta-file", type=Path, default=DEFAULT_META_FILE)
    parser.add_argument("--copy", action="store_true", help="使用真实复制；默认优先硬链接以节省空间")
    parser.add_argument("--force", action="store_true", help="删除并重建输出目录")
    parser.add_argument("--include-non-pet", action="store_true", help="保留 Stanford 中的野生犬科类别")
    args = parser.parse_args()

    out_dir = args.out_dir.resolve()
    if args.force:
        shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_hardlink = not args.copy
    oxford_stats = collect_oxford(args.oxford_dir.resolve(), out_dir, use_hardlink)
    stanford_grouped = collect_stanford(args.stanford_dir.resolve(), args.include_non_pet)
    stanford_stats = add_stanford(stanford_grouped, out_dir, use_hardlink)
    kaggle_cat_grouped = collect_kaggle_cats(args.kaggle_cats_dir.resolve())
    kaggle_cat_stats = add_kaggle_cats(kaggle_cat_grouped, out_dir, use_hardlink)
    classes, class_type = build_meta(out_dir, args.meta_file.resolve())
    split_counts = count_images(out_dir)

    duplicate_classes = sorted(set(oxford_stats) & set(stanford_stats))
    duplicate_cats = sorted(set(oxford_stats) & set(kaggle_cat_stats))
    new_cats = sorted(set(kaggle_cat_stats) - set(oxford_stats))
    print("\n扩展数据集构建完成")
    print(f"输出目录：{out_dir}")
    print(f"类别数：{len(classes)}（猫 {sum(1 for t in class_type.values() if t == 'cat')}，狗 {sum(1 for t in class_type.values() if t == 'dog')}）")
    print(f"图片数：train={split_counts['train']} val={split_counts['val']} test={split_counts['test']}")
    print(f"与原 Oxford 合并的重复/同义狗品种数：{len(duplicate_classes)}")
    if duplicate_classes:
        print("重复/同义类别：" + ", ".join(duplicate_classes[:40]))
    print(f"与原 Oxford 合并的重复/同义猫品种数：{len(duplicate_cats)}")
    if duplicate_cats:
        print("重复/同义猫类别：" + ", ".join(duplicate_cats[:40]))
    print(f"新增猫品种数：{len(new_cats)}")
    if new_cats:
        print("新增猫类别：" + ", ".join(new_cats[:40]))
    print(f"类别元数据：{args.meta_file.resolve()}")
    print("\n训练示例：")
    print("python 03_train.py --data-dir dataset_extended --model-dir models_extended")


if __name__ == "__main__":
    main()
