"""
统一 AI 推理服务：FastAPI

提供两类能力，供 Spring Boot 后端调用：
1. 品种识别 1.1（140 类，默认）：POST /api/recognize
2. PetFace-ID 2.0 个体识别：POST /api/petface/embed, POST /api/petface/verify

运行：
    python inference_server.py

如需切回 1.0 模型（37 类），可手动指定：
    PET_BREED_MODEL_DIR=models python inference_server.py
"""

import io
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import onnxruntime as ort
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from petface.petface_model import (
    ModelConfig,
    build_transform as build_petface_transform,
    cosine_similarity,
    embed_image,
    load_model as load_petface_model,
)

BASE_DIR  = Path(__file__).parent
MODEL_DIR = Path(os.environ.get("PET_BREED_MODEL_DIR", BASE_DIR / "models_extended")).expanduser().resolve()
ONNX_PATH = MODEL_DIR / "pet_classifier.onnx"
META_FILE = MODEL_DIR / "class_meta.json"

PETFACE_MODEL_DIR = Path(os.environ.get("PETFACE_MODEL_DIR", BASE_DIR / "models" / "petface")).expanduser().resolve()
PETFACE_CHECKPOINT = Path(os.environ.get("PETFACE_CHECKPOINT", PETFACE_MODEL_DIR / "petface_id_best.pth")).expanduser().resolve()
PETFACE_META_FILE = Path(os.environ.get("PETFACE_META", PETFACE_MODEL_DIR / "petface_meta.json")).expanduser().resolve()
PETFACE_THRESHOLD = float(os.environ.get("PETFACE_THRESHOLD", "0.75"))

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


def _load_petface_meta() -> dict[str, Any]:
    if PETFACE_META_FILE.exists():
        with open(PETFACE_META_FILE, encoding="utf-8") as f:
            return json.load(f)
    return {
        "model_version": "PetFace-ID-2.0",
        "model_name": "hf-hub:BVRA/MegaDescriptor-B-224",
        "img_size": 224,
        "embed_dim": 512,
    }


PETFACE_META = _load_petface_meta()
PETFACE_MODEL = None
PETFACE_DEVICE = None
PETFACE_TRANSFORM = None
PETFACE_LOAD_ERROR: str | None = None

if PETFACE_CHECKPOINT.exists():
    try:
        PETFACE_MODEL, PETFACE_DEVICE = load_petface_model(
            ModelConfig(
                model_name=os.environ.get("PETFACE_MODEL_NAME", PETFACE_META.get("model_name", "hf-hub:BVRA/MegaDescriptor-B-224")),
                img_size=int(os.environ.get("PETFACE_IMG_SIZE", PETFACE_META.get("img_size", 224))),
                embed_dim=int(os.environ.get("PETFACE_EMBED_DIM", PETFACE_META.get("embed_dim", 512))),
                checkpoint=str(PETFACE_CHECKPOINT),
            )
        )
        PETFACE_TRANSFORM = build_petface_transform(int(os.environ.get("PETFACE_IMG_SIZE", PETFACE_META.get("img_size", 224))))
    except Exception as exc:  # noqa: BLE001 - health endpoint exposes this for deployment diagnosis.
        PETFACE_LOAD_ERROR = str(exc)
else:
    PETFACE_LOAD_ERROR = f"未找到 PetFace 模型：{PETFACE_CHECKPOINT}"

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
app = FastAPI(title="宠物统一 AI 识别服务", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def preprocess_breed_image(image_bytes: bytes) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    return arr.transpose(2, 0, 1)[np.newaxis]       # (1, 3, H, W)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


async def read_image_upload(file: UploadFile) -> Image.Image:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件（jpg 或 png）")
    data = await file.read()
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="图片读取失败，请确认文件格式正确") from exc


async def read_image_bytes(file: UploadFile) -> bytes:
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="请上传图片文件（jpg 或 png）")
    return await file.read()


def ensure_petface_loaded() -> None:
    if PETFACE_MODEL is None or PETFACE_DEVICE is None or PETFACE_TRANSFORM is None:
        raise HTTPException(
            status_code=503,
            detail=f"PetFace 2.0 模型未加载：{PETFACE_LOAD_ERROR or 'unknown error'}",
        )


def petface_confidence_level(similarity: float) -> str:
    if similarity >= PETFACE_THRESHOLD:
        return "high"
    if similarity >= 0.60:
        return "medium"
    return "low"


@app.post("/api/recognize")
async def recognize(file: UploadFile = File(..., description="宠物图片（jpg/png）")):
    data    = await read_image_bytes(file)
    inp     = preprocess_breed_image(data)
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


@app.post("/api/petface/embed")
async def petface_embed(file: UploadFile = File(..., description="宠物图片（jpg/png）")):
    ensure_petface_loaded()
    image = await read_image_upload(file)
    emb = embed_image(PETFACE_MODEL, image, PETFACE_TRANSFORM, PETFACE_DEVICE)
    return {
        "code": 200,
        "message": "特征提取成功",
        "data": {
            "model_version": PETFACE_META.get("model_version", "PetFace-ID-2.0"),
            "model_name": PETFACE_META.get("model_name", "hf-hub:BVRA/MegaDescriptor-B-224"),
            "embedding_dim": int(emb.shape[0]),
            "embedding": emb.round(6).tolist(),
            "norm": round(float(np.linalg.norm(emb)), 6),
        },
    }


@app.post("/api/petface/verify")
async def petface_verify(
    file1: UploadFile = File(..., description="第一张宠物图片（jpg/png）"),
    file2: UploadFile = File(..., description="第二张宠物图片（jpg/png）"),
):
    ensure_petface_loaded()
    image1 = await read_image_upload(file1)
    image2 = await read_image_upload(file2)
    emb1 = embed_image(PETFACE_MODEL, image1, PETFACE_TRANSFORM, PETFACE_DEVICE)
    emb2 = embed_image(PETFACE_MODEL, image2, PETFACE_TRANSFORM, PETFACE_DEVICE)
    similarity = cosine_similarity(emb1, emb2)
    return {
        "code": 200,
        "message": "验证完成",
        "data": {
            "same_pet": similarity >= PETFACE_THRESHOLD,
            "similarity": round(similarity, 4),
            "threshold": PETFACE_THRESHOLD,
            "confidence_level": petface_confidence_level(similarity),
            "model_version": PETFACE_META.get("model_version", "PetFace-ID-2.0"),
        },
    }


@app.get("/health")
def health():
    return {
        "status": "ok",
        "breed": {
            "loaded": True,
            "model": ONNX_PATH.name,
            "num_classes": len(IDX_TO_CLASS),
        },
        "petface": {
            "loaded": PETFACE_MODEL is not None,
            "model": PETFACE_META.get("model_name", "hf-hub:BVRA/MegaDescriptor-B-224"),
            "model_version": PETFACE_META.get("model_version", "PetFace-ID-2.0"),
            "embedding_dim": PETFACE_META.get("embed_dim", 512),
            "checkpoint": str(PETFACE_CHECKPOINT),
            "device": str(PETFACE_DEVICE) if PETFACE_DEVICE is not None else None,
            "error": PETFACE_LOAD_ERROR,
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
