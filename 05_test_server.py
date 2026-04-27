"""
步骤 5：测试推理服务是否正常工作
请先启动服务：python inference_server.py

运行：python 05_test_server.py [图片路径]
示例：python 05_test_server.py ./sample.jpg
"""

import sys
import json
import requests
from pathlib import Path

SERVER = "http://localhost:8000"


def test_health():
    resp = requests.get(f"{SERVER}/health", timeout=3)
    print("服务健康检查：", resp.json())


def recognize(image_path: str):
    p = Path(image_path)
    if not p.exists():
        print(f"[错误] 文件不存在：{image_path}")
        return

    with open(p, "rb") as f:
        files = {"file": (p.name, f, "image/jpeg")}
        resp = requests.post(f"{SERVER}/api/recognize", files=files, timeout=10)

    result = resp.json()
    if result.get("code") != 200:
        print("识别失败：", result)
        return

    data = result["data"]
    print(f"\n识别结果：")
    print(f"  物种：{data['pet_type_cn']}（{data['pet_type']}）")
    print(f"  品种：{data['breed_cn']}（{data['breed']}）")
    print(f"  置信度：{data['confidence']*100:.1f}%")
    print(f"\nTop-5 候选：")
    for i, item in enumerate(data["top5"], 1):
        bar = "█" * int(item["confidence"] * 30)
        print(f"  {i}. {item['breed_cn']:12s}  {bar:30s}  {item['confidence']*100:5.1f}%")


def test_with_url(image_url: str):
    """通过 URL 下载图片并识别（测试用）"""
    import io
    import urllib.request

    print(f"下载测试图片：{image_url}")
    with urllib.request.urlopen(image_url) as r:
        img_bytes = r.read()

    files = {"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    resp = requests.post(f"{SERVER}/api/recognize", files=files, timeout=10)

    result = resp.json()
    data = result["data"]
    print(f"\n识别结果：{data['pet_type_cn']} - {data['breed_cn']}（置信度 {data['confidence']*100:.1f}%）")


if __name__ == "__main__":
    try:
        test_health()
    except requests.exceptions.ConnectionError:
        print("[错误] 无法连接到推理服务，请先运行：python inference_server.py")
        sys.exit(1)

    if len(sys.argv) > 1:
        recognize(sys.argv[1])
    else:
        print("\n用法：python 05_test_server.py <图片路径>")
        print("示例：python 05_test_server.py /path/to/cat.jpg")
