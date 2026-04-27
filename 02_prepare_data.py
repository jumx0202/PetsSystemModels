"""
步骤 2：整理数据集为 train / val / test 三份
Oxford-IIIT Pet Dataset 命名规则：
  - 猫（12种）：首字母大写，如 Abyssinian_1.jpg
  - 狗（25种）：全部小写，如 american_bulldog_1.jpg
运行：python 02_prepare_data.py
"""

import json
import random
import shutil
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm

BASE_DIR  = Path(__file__).parent
SRC_DIR   = BASE_DIR / "data_raw" / "images"
DEST_DIR  = BASE_DIR / "dataset"
META_FILE = BASE_DIR / "models" / "class_meta.json"

SPLIT_RATIO = (0.8, 0.1, 0.1)   # train / val / test
RANDOM_SEED = 42

# Oxford-IIIT Pet Dataset 中的猫品种（首字母大写）
CAT_BREEDS = {
    "Abyssinian", "Bengal", "Birman", "Bombay", "British_Shorthair",
    "Egyptian_Mau", "Maine_Coon", "Persian", "Ragdoll",
    "Russian_Blue", "Siamese", "Sphynx",
}


def parse_class(filename: str) -> str:
    """从文件名提取类别名，去掉末尾的数字编号。"""
    stem = Path(filename).stem          # e.g. "Abyssinian_100"
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]                 # e.g. "Abyssinian"
    return stem


def main():
    random.seed(RANDOM_SEED)

    if not SRC_DIR.exists():
        print(f"[错误] 未找到原始图片目录：{SRC_DIR}")
        print("请先运行 python 01_download_dataset.py")
        return

    # ── 按类别分组 ─────────────────────────────────────
    class_images: dict[str, list[Path]] = defaultdict(list)
    for img in SRC_DIR.glob("*.jpg"):
        cls = parse_class(img.name)
        class_images[cls].append(img)

    classes = sorted(class_images.keys())
    print(f"共发现 {len(classes)} 个类别，{sum(len(v) for v in class_images.values())} 张图片")

    # ── 构建类别元数据 ─────────────────────────────────
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    idx_to_class = {i: cls for cls, i in class_to_idx.items()}
    class_type   = {cls: ("cat" if cls in CAT_BREEDS else "dog") for cls in classes}

    META_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "class_to_idx": class_to_idx,
            "idx_to_class": {str(k): v for k, v in idx_to_class.items()},
            "class_type":   class_type,
            "num_classes":  len(classes),
        }, f, ensure_ascii=False, indent=2)
    print(f"类别元数据已保存：{META_FILE}")

    # ── 分割并复制到 dataset/ ──────────────────────────
    for split in ("train", "val", "test"):
        shutil.rmtree(DEST_DIR / split, ignore_errors=True)

    total_copied = {"train": 0, "val": 0, "test": 0}

    for cls, imgs in tqdm(class_images.items(), desc="整理类别", unit="类"):
        random.shuffle(imgs)
        n = len(imgs)
        t_end = int(n * SPLIT_RATIO[0])
        v_end = t_end + int(n * SPLIT_RATIO[1])

        splits = {
            "train": imgs[:t_end],
            "val":   imgs[t_end:v_end],
            "test":  imgs[v_end:],
        }

        for split, files in splits.items():
            dest = DEST_DIR / split / cls
            dest.mkdir(parents=True, exist_ok=True)
            for f in files:
                shutil.copy(f, dest / f.name)
            total_copied[split] += len(files)

    # ── 统计结果 ──────────────────────────────────────
    print("\n数据集整理完成：")
    for split, cnt in total_copied.items():
        print(f"  {split:5s}: {cnt} 张")
    print(f"\n猫的品种（{sum(1 for t in class_type.values() if t=='cat')} 种）：")
    print("  " + ", ".join(c for c in classes if class_type[c] == "cat"))
    print(f"\n狗的品种（{sum(1 for t in class_type.values() if t=='dog')} 种）：")
    print("  " + ", ".join(c for c in classes if class_type[c] == "dog"))
    print("\n下一步：运行 python 03_train.py")


if __name__ == "__main__":
    main()
