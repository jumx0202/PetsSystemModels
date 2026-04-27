"""
推理服务：FastAPI + ONNX Runtime
提供宠物品种识别 HTTP 接口，供 Spring Boot 后端调用

运行：python inference_server.py
接口：POST http://localhost:8000/api/recognize  (form-data, field: file)
"""

import io
import json
from pathlib import Path

import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
ONNX_PATH = MODEL_DIR / "pet_classifier.onnx"
META_FILE = MODEL_DIR / "class_meta.json"

IMG_SIZE = 380
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# ── 启动时加载模型 ──────────────────────────────────────
if not ONNX_PATH.exists():
    raise FileNotFoundError(
        f"未找到 ONNX 模型：{ONNX_PATH}\n请先运行 python 04_export_onnx.py"
    )

SESSION = ort.InferenceSession(str(ONNX_PATH))

with open(META_FILE, encoding="utf-8") as f:
    _meta = json.load(f)

IDX_TO_CLASS: dict[int, str] = {int(k): v for k, v in _meta["idx_to_class"].items()}
CLASS_TYPE:   dict[str, str] = _meta["class_type"]

# 中文品种名映射（论文展示友好）
BREED_CN: dict[str, str] = {
    "Abyssinian":              "阿比西尼亚猫",
    "Bengal":                  "孟加拉猫",
    "Birman":                  "伯曼猫",
    "Bombay":                  "孟买猫",
    "British_Shorthair":       "英国短毛猫",
    "Egyptian_Mau":            "埃及猫",
    "Maine_Coon":              "缅因猫",
    "Persian":                 "波斯猫",
    "Ragdoll":                 "布偶猫",
    "Russian_Blue":            "俄罗斯蓝猫",
    "Siamese":                 "暹罗猫",
    "Sphynx":                  "斯芬克斯猫",
    "american_bulldog":        "美国斗牛犬",
    "american_pit_bull_terrier": "美国比特犬",
    "basset_hound":            "巴吉度猎犬",
    "beagle":                  "比格猎犬",
    "boxer":                   "拳师犬",
    "chihuahua":               "吉娃娃",
    "english_cocker_spaniel":  "英国可卡犬",
    "english_setter":          "英国雪达犬",
    "german_shorthaired":      "德国短毛指示犬",
    "great_pyrenees":          "大白熊犬",
    "havanese":                "哈瓦那犬",
    "japanese_chin":           "日本狆",
    "keeshond":                "荷兰毛狮犬",
    "leonberger":              "雷昂贝格犬",
    "miniature_pinscher":      "迷你杜宾犬",
    "newfoundland":            "纽芬兰犬",
    "pomeranian":              "博美犬",
    "pug":                     "巴哥犬",
    "saint_bernard":           "圣伯纳犬",
    "samoyed":                 "萨摩耶",
    "scottish_terrier":        "苏格兰梗",
    "shiba_inu":               "柴犬",
    "staffordshire_bull_terrier": "斯塔福郡斗牛梗",
    "wheaten_terrier":         "软毛麦色梗",
    "yorkshire_terrier":       "约克夏梗",
}

# ── FastAPI 应用 ────────────────────────────────────────
app = FastAPI(title="宠物识别服务", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis]       # (1, 3, H, W)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(..., description="宠物图片（jpg/png）")):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件（jpg 或 png）")

    data    = await file.read()
    inp     = preprocess(data)
    logits  = SESSION.run(["logits"], {"image": inp})[0][0]
    probs   = softmax(logits)

    top5_idx = probs.argsort()[-5:][::-1]
    top5 = [
        {
            "breed":      IDX_TO_CLASS[int(i)],
            "breed_cn":   BREED_CN.get(IDX_TO_CLASS[int(i)], IDX_TO_CLASS[int(i)]),
            "confidence": round(float(probs[i]), 4),
        }
        for i in top5_idx
    ]

    best      = top5[0]
    pet_type  = CLASS_TYPE.get(best["breed"], "unknown")
    pet_type_cn = "猫" if pet_type == "cat" else "狗"

    return {
        "code":    200,
        "message": "识别成功",
        "data": {
            "pet_type":    pet_type,
            "pet_type_cn": pet_type_cn,
            "breed":       best["breed"],
            "breed_cn":    best["breed_cn"],
            "confidence":  best["confidence"],
            "top5":        top5,
        },
    }


@app.get("/health")
def health():
    return {
        "status":      "ok",
        "model":       ONNX_PATH.name,
        "num_classes": len(IDX_TO_CLASS),
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
