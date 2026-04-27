"""
步骤 1：下载 Oxford-IIIT Pet Dataset
  - 图像包：~800MB
  - 标注包：~19MB
运行：python 01_download_dataset.py
"""

import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data_raw"
DATA_DIR.mkdir(exist_ok=True)

URLS = {
    "images.tar.gz":      "https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz",
    "annotations.tar.gz": "https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz",
}

CHUNK_SIZE  = 1024 * 1024   # 1MB 分块
MAX_RETRIES = 10
RETRY_WAIT  = 5             # 秒


def download(url: str, dest: Path):
    if dest.exists():
        size = dest.stat().st_size
        print(f"  已存在（{size/1024/1024:.0f} MB），跳过：{dest.name}")
        return

    print(f"  下载 {dest.name}")
    tmp = dest.with_suffix(".tmp")

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            # 支持从上次中断处继续（若 .tmp 存在）
            downloaded = tmp.stat().st_size if tmp.exists() else 0
            headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}

            resp = requests.get(url, headers=headers, stream=True, timeout=30)

            # 206 = 断点续传成功；200 = 服务器不支持，从头下
            if resp.status_code == 200 and downloaded:
                downloaded = 0
                tmp.unlink(missing_ok=True)
            elif resp.status_code not in (200, 206):
                raise RuntimeError(f"HTTP {resp.status_code}")

            total = int(resp.headers.get("content-length", 0)) + downloaded

            mode = "ab" if downloaded else "wb"
            with open(tmp, mode) as f, tqdm(
                total=total, initial=downloaded,
                unit="B", unit_scale=True, desc=dest.name
            ) as bar:
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))

            tmp.rename(dest)
            print(f"  ✓ {dest.name} 下载完成")
            return

        except Exception as e:
            print(f"  [尝试 {attempt}/{MAX_RETRIES}] 出错：{e}")
            if attempt < MAX_RETRIES:
                print(f"  {RETRY_WAIT}s 后重试...")
                time.sleep(RETRY_WAIT)

    raise RuntimeError(f"下载失败（已重试 {MAX_RETRIES} 次）：{url}")


def extract(archive: Path, dest: Path):
    flag = dest / ".extracted"
    if flag.exists():
        print(f"  已解压，跳过：{archive.name}")
        return
    print(f"  解压 {archive.name} → {dest}/")
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in tqdm(members, desc="解压", unit="文件"):
            tar.extract(member, dest)
    flag.touch()


def main():
    print("=" * 50)
    print("Oxford-IIIT Pet Dataset 下载工具")
    print("=" * 50)

    for filename, url in URLS.items():
        archive = DATA_DIR / filename
        download(url, archive)
        extract(archive, DATA_DIR)

    images_dir = DATA_DIR / "images"
    jpg_count = len(list(images_dir.glob("*.jpg")))
    print(f"\n下载完成！共 {jpg_count} 张图片在 {images_dir}")
    print("下一步：运行 python 02_prepare_data.py")


if __name__ == "__main__":
    main()
