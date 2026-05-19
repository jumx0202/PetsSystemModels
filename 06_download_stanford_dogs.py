"""
下载 Stanford Dogs Dataset，用于扩充 1.0 宠物品种识别数据集。

数据来源：
  http://vision.stanford.edu/aditya86/ImageNetDogs/

运行：
  python 06_download_stanford_dogs.py
"""

import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DEST_DIR = BASE_DIR / "data_raw" / "stanford_dogs"
DEST_DIR.mkdir(parents=True, exist_ok=True)

URL = "http://vision.stanford.edu/aditya86/ImageNetDogs/images.tar"
ARCHIVE = DEST_DIR / "images.tar"
IMAGES_DIR = DEST_DIR / "Images"

CHUNK_SIZE = 1024 * 1024
MAX_RETRIES = 10
RETRY_WAIT = 5


def download(url: str, dest: Path):
    if dest.exists():
        print(f"已存在，跳过下载：{dest} ({dest.stat().st_size / 1024 / 1024:.0f} MB)")
        return

    tmp = dest.with_suffix(dest.suffix + ".tmp")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            downloaded = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
            resp = requests.get(url, headers=headers, stream=True, timeout=60)

            if resp.status_code == 200 and downloaded:
                tmp.unlink(missing_ok=True)
                downloaded = 0
            elif resp.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {resp.status_code}")

            total = int(resp.headers.get("content-length", 0)) + downloaded
            mode = "ab" if downloaded else "wb"
            with open(tmp, mode) as f, tqdm(
                total=total,
                initial=downloaded,
                unit="B",
                unit_scale=True,
                desc=dest.name,
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            tmp.rename(dest)
            print(f"下载完成：{dest}")
            return
        except Exception as exc:
            print(f"[尝试 {attempt}/{MAX_RETRIES}] 下载失败：{exc}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_WAIT)

    raise RuntimeError(f"下载失败：{url}")


def extract(archive: Path, dest: Path):
    flag = dest / ".stanford_dogs_extracted"
    if flag.exists() and IMAGES_DIR.exists():
        print(f"已解压，跳过：{IMAGES_DIR}")
        return

    print(f"开始解压：{archive}")
    with tarfile.open(archive, "r") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="解压", unit="文件"):
            tar.extract(member, dest)
    flag.touch()
    print(f"解压完成：{IMAGES_DIR}")


def main():
    print("=" * 60)
    print("Stanford Dogs Dataset 下载工具")
    print("=" * 60)
    download(URL, ARCHIVE)
    extract(ARCHIVE, DEST_DIR)
    class_dirs = [p for p in IMAGES_DIR.iterdir() if p.is_dir()] if IMAGES_DIR.exists() else []
    image_count = len(list(IMAGES_DIR.rglob("*.jpg"))) if IMAGES_DIR.exists() else 0
    print(f"类别数：{len(class_dirs)}")
    print(f"图片数：{image_count}")
    print("下一步：python 07_build_extended_dataset.py")


if __name__ == "__main__":
    main()
