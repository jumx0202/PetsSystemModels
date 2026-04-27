"""
步骤 4：将训练好的 .pth 模型导出为 ONNX 格式（用于推理服务）
注意：ONNX 导出必须在 CPU 上进行

运行：python 04_export_onnx.py
"""

import json
from pathlib import Path

import torch
import timm
import onnx
import onnxruntime as ort
import numpy as np

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
META_FILE = MODEL_DIR / "class_meta.json"
CKPT_PATH = MODEL_DIR / "pet_classifier_best.pth"
ONNX_PATH = MODEL_DIR / "pet_classifier.onnx"

MODEL_NAME = "efficientnet_b4"
IMG_SIZE   = 380


def export():
    if not CKPT_PATH.exists():
        print(f"[错误] 未找到训练好的模型：{CKPT_PATH}")
        print("请先运行 python 03_train.py")
        return

    with open(META_FILE) as f:
        meta = json.load(f)
    num_classes = meta["num_classes"]
    print(f"类别数：{num_classes}")

    # ── 加载模型到 CPU ──────────────────────────────────
    model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=num_classes)
    state = torch.load(CKPT_PATH, map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    print("模型权重已加载")

    # ── 导出 ONNX ──────────────────────────────────────
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    torch.onnx.export(
        model,
        dummy,
        str(ONNX_PATH),
        input_names=["image"],
        output_names=["logits"],
        dynamic_axes={"image": {0: "batch_size"}, "logits": {0: "batch_size"}},
        opset_version=17,
        do_constant_folding=True,
    )
    print(f"ONNX 模型已导出：{ONNX_PATH}")

    # ── 验证 ONNX 结构 ──────────────────────────────────
    onnx_model = onnx.load(str(ONNX_PATH))
    onnx.checker.check_model(onnx_model)
    print("ONNX 结构验证通过")

    # ── 对比 PyTorch 与 ONNX 输出 ──────────────────────
    session = ort.InferenceSession(str(ONNX_PATH))
    np_input = dummy.numpy()

    with torch.no_grad():
        pt_out = model(dummy).numpy()
    ort_out = session.run(["logits"], {"image": np_input})[0]

    max_diff = np.abs(pt_out - ort_out).max()
    print(f"PyTorch vs ONNX 最大误差：{max_diff:.6f}（< 1e-4 表示正常）")

    size_mb = ONNX_PATH.stat().st_size / 1024 / 1024
    print(f"ONNX 文件大小：{size_mb:.1f} MB")
    print(f"\n导出完成！下一步：python inference_server.py")


if __name__ == "__main__":
    export()
