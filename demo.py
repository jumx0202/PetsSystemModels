"""
宠物品种识别 Demo（Flask + 内嵌 HTML）
直接加载 .pth 权重，无需先导出 ONNX

运行：/Users/org/miniconda3/bin/python demo.py
访问：http://127.0.0.1:7860
"""

import io
import json
import base64
from pathlib import Path

import torch
import timm
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template_string

# ── 路径 ──────────────────────────────────────────────
BASE_DIR  = Path(__file__).parent
MODEL_DIR = BASE_DIR / "models"
CKPT_BEST = MODEL_DIR / "pet_classifier_best.pth"
META_FILE = MODEL_DIR / "class_meta.json"

# ── 模型参数 ──────────────────────────────────────────
MODEL_NAME = "efficientnet_b4"
IMG_SIZE   = 380

BREED_CN = {
    "Abyssinian":       "阿比西尼亚猫",
    "Bengal":           "孟加拉猫",
    "Birman":           "伯曼猫",
    "Bombay":           "孟买猫",
    "British_Shorthair":"英国短毛猫",
    "Egyptian_Mau":     "埃及猫",
    "Maine_Coon":       "缅因猫",
    "Persian":          "波斯猫",
    "Ragdoll":          "布偶猫",
    "Russian_Blue":     "俄罗斯蓝猫",
    "Siamese":          "暹罗猫",
    "Sphynx":           "斯芬克斯猫",
    "american_bulldog":           "美国斗牛犬",
    "american_pit_bull_terrier":  "美国比特犬",
    "basset_hound":               "巴吉度猎犬",
    "beagle":                     "比格犬",
    "boxer":                      "拳师犬",
    "chihuahua":                  "吉娃娃",
    "english_cocker_spaniel":     "英国可卡犬",
    "english_setter":             "英国雪达犬",
    "german_shorthaired":         "德国短毛猎犬",
    "great_pyrenees":             "大白熊犬",
    "havanese":                   "哈瓦那犬",
    "japanese_chin":              "日本狆",
    "keeshond":                   "荷兰毛狮犬",
    "leonberger":                 "莱昂贝格犬",
    "miniature_pinscher":         "迷你杜宾犬",
    "newfoundland":               "纽芬兰犬",
    "pomeranian":                 "博美犬",
    "pug":                        "巴哥犬",
    "saint_bernard":              "圣伯纳犬",
    "samoyed":                    "萨摩耶",
    "scottish_terrier":           "苏格兰梗",
    "shiba_inu":                  "柴犬",
    "staffordshire_bull_terrier": "斯塔福斗牛梗",
    "wheaten_terrier":            "软毛麦色梗",
    "yorkshire_terrier":          "约克夏梗",
}

# ── 设备 ─────────────────────────────────────────────
if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("✓ 使用 Apple MPS 加速")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# ── 加载元数据 ────────────────────────────────────────
with open(META_FILE, encoding="utf-8") as f:
    meta = json.load(f)
IDX_TO_CLASS = {int(k): v for k, v in meta["idx_to_class"].items()}
CLASS_TYPE   = meta["class_type"]
NUM_CLASSES  = meta["num_classes"]

# ── 加载模型 ──────────────────────────────────────────
print("正在加载模型...")
model = timm.create_model(MODEL_NAME, pretrained=False, num_classes=NUM_CLASSES)
model.load_state_dict(torch.load(CKPT_BEST, map_location=DEVICE))
model = model.to(DEVICE)
model.eval()
print(f"✓ 模型加载完成（{NUM_CLASSES} 个品种）")

preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ── Flask ─────────────────────────────────────────────
app = Flask(__name__)

HTML = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>🐾 宠物品种识别</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
         background: #f5f7fa; color: #333; min-height: 100vh; }
  .header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white; padding: 24px; text-align: center; }
  .header h1 { font-size: 28px; margin-bottom: 6px; }
  .header p  { opacity: 0.85; font-size: 14px; }
  .container { max-width: 860px; margin: 32px auto; padding: 0 16px; }
  .card { background: white; border-radius: 16px; padding: 28px;
          box-shadow: 0 4px 20px rgba(0,0,0,0.08); margin-bottom: 20px; }

  /* 上传区 */
  .upload-area { border: 2px dashed #c0c8e0; border-radius: 12px;
                 padding: 40px 20px; text-align: center; cursor: pointer;
                 transition: all 0.2s; background: #fafbff; }
  .upload-area:hover, .upload-area.drag-over { border-color: #667eea; background: #f0f2ff; }
  .upload-icon { font-size: 48px; margin-bottom: 12px; }
  .upload-area p { color: #666; font-size: 14px; margin-top: 8px; }
  #fileInput { display: none; }
  #preview { max-width: 100%; max-height: 320px; border-radius: 10px;
             margin-top: 16px; display: none; object-fit: contain; }
  .btn { background: linear-gradient(135deg, #667eea, #764ba2);
         color: white; border: none; padding: 12px 36px; border-radius: 8px;
         font-size: 16px; cursor: pointer; margin-top: 16px;
         transition: opacity 0.2s; }
  .btn:hover { opacity: 0.88; }
  .btn:disabled { opacity: 0.5; cursor: not-allowed; }

  /* 结果 */
  #result { display: none; }
  .result-header { display: flex; align-items: center; gap: 16px; margin-bottom: 20px; }
  .pet-badge { font-size: 40px; }
  .breed-name { font-size: 22px; font-weight: 700; }
  .breed-en   { font-size: 13px; color: #888; margin-top: 2px; }
  .confidence-big { font-size: 32px; font-weight: 800;
                    background: linear-gradient(135deg, #667eea, #764ba2);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent; }
  .conf-label { font-size: 12px; color: #aaa; margin-top: 2px; }

  .top5-title { font-size: 14px; font-weight: 600; color: #555;
                margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px; }
  .bar-row { margin-bottom: 10px; }
  .bar-label { display: flex; justify-content: space-between; margin-bottom: 3px;
               font-size: 13px; }
  .bar-track { background: #eef0f8; border-radius: 4px; height: 10px; overflow: hidden; }
  .bar-fill  { height: 100%; border-radius: 4px;
               background: linear-gradient(90deg, #667eea, #764ba2);
               transition: width 0.6s ease; }
  .bar-row.first .bar-fill { background: linear-gradient(90deg, #f7971e, #ffd200); }

  .loading { text-align: center; padding: 20px; color: #888; display: none; }
  .spinner { width: 36px; height: 36px; border: 3px solid #e0e0e0;
             border-top-color: #667eea; border-radius: 50%;
             animation: spin 0.8s linear infinite; margin: 0 auto 10px; }
  @keyframes spin { to { transform: rotate(360deg); } }
</style>
</head>
<body>
<div class="header">
  <h1>🐾 宠物品种识别</h1>
  <p>EfficientNet-B4 · 37 个品种（12 猫 + 25 狗）· 测试集准确率 96.1%</p>
</div>

<div class="container">
  <div class="card">
    <div class="upload-area" id="uploadArea" onclick="document.getElementById('fileInput').click()">
      <div class="upload-icon">📷</div>
      <strong>点击上传或拖拽图片到此处</strong>
      <p>支持 JPG / PNG，建议使用宠物正面清晰照</p>
      <img id="preview">
    </div>
    <input type="file" id="fileInput" accept="image/*">
    <div style="text-align:center">
      <button class="btn" id="predictBtn" onclick="predict()" disabled>开始识别</button>
    </div>
  </div>

  <div class="loading" id="loading">
    <div class="spinner"></div>
    <p>识别中，请稍候…</p>
  </div>

  <div class="card" id="result">
    <div class="result-header">
      <div class="pet-badge" id="petBadge"></div>
      <div>
        <div class="breed-name" id="breedCn"></div>
        <div class="breed-en"  id="breedEn"></div>
      </div>
      <div style="margin-left:auto; text-align:right">
        <div class="confidence-big" id="confBig"></div>
        <div class="conf-label">置信度</div>
      </div>
    </div>
    <div class="top5-title">Top-5 品种概率</div>
    <div id="bars"></div>
  </div>
</div>

<script>
const fileInput   = document.getElementById('fileInput');
const preview     = document.getElementById('preview');
const predictBtn  = document.getElementById('predictBtn');
const uploadArea  = document.getElementById('uploadArea');

fileInput.addEventListener('change', e => {
  const file = e.target.files[0];
  if (!file) return;
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  predictBtn.disabled = false;
});

uploadArea.addEventListener('dragover', e => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
uploadArea.addEventListener('drop', e => {
  e.preventDefault();
  uploadArea.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (!file || !file.type.startsWith('image/')) return;
  fileInput.files = e.dataTransfer.files;
  preview.src = URL.createObjectURL(file);
  preview.style.display = 'block';
  predictBtn.disabled = false;
});

async function predict() {
  const file = fileInput.files[0];
  if (!file) return;

  document.getElementById('result').style.display  = 'none';
  document.getElementById('loading').style.display = 'block';
  predictBtn.disabled = true;

  const fd = new FormData();
  fd.append('file', file);

  try {
    const resp = await fetch('/predict', { method: 'POST', body: fd });
    const data = await resp.json();

    document.getElementById('petBadge').textContent = data.pet_type === 'cat' ? '🐱' : '🐶';
    document.getElementById('breedCn').textContent  = data.breed_cn;
    document.getElementById('breedEn').textContent  = data.breed;
    document.getElementById('confBig').textContent  = (data.confidence * 100).toFixed(1) + '%';

    const barsDiv = document.getElementById('bars');
    barsDiv.innerHTML = '';
    data.top5.forEach((item, i) => {
      const pct = (item.confidence * 100).toFixed(1);
      barsDiv.innerHTML += `
        <div class="bar-row ${i === 0 ? 'first' : ''}">
          <div class="bar-label">
            <span>${item.breed_cn}<span style="color:#aaa;font-size:11px"> ${item.breed}</span></span>
            <span>${pct}%</span>
          </div>
          <div class="bar-track"><div class="bar-fill" style="width:${pct}%"></div></div>
        </div>`;
    });

    document.getElementById('loading').style.display = 'none';
    document.getElementById('result').style.display  = 'block';
  } catch(err) {
    document.getElementById('loading').style.display = 'none';
    alert('识别失败：' + err);
  }
  predictBtn.disabled = false;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "no file"}), 400

    data = request.files["file"].read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")
    tensor = preprocess(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs  = torch.softmax(logits, dim=1)[0]

    top5_probs, top5_idx = probs.topk(5)
    top5 = []
    for prob, idx in zip(top5_probs.cpu().tolist(), top5_idx.cpu().tolist()):
        breed = IDX_TO_CLASS[idx]
        top5.append({
            "breed":      breed,
            "breed_cn":   BREED_CN.get(breed, breed),
            "confidence": round(prob, 4),
        })

    best = top5[0]
    return jsonify({
        "breed":      best["breed"],
        "breed_cn":   best["breed_cn"],
        "confidence": best["confidence"],
        "pet_type":   CLASS_TYPE[best["breed"]],
        "top5":       top5,
    })


if __name__ == "__main__":
    print("访问：http://127.0.0.1:7860")
    app.run(host="127.0.0.1", port=7860, debug=False)
