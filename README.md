# AI 统一识别模块

本仓库是宠物管理系统的 Python AI 服务端，当前已经从单一“宠物品种识别 1.0”升级为统一 FastAPI 推理服务，同时承载：

- **品种识别 1.0**：Oxford-IIIT Pet，37 类猫狗品种。
- **品种识别 1.1 增强版**：扩展到 140 类，其中猫 20 类、狗 120 类。
- **PetFace-ID 2.0 个体识别**：输出 512 维宠物个体特征，用于同宠验证与 AI 寻宠相似检索。

Spring Boot 后端通过 HTTP 调用本服务，前端不直接访问 Python 服务。

## 目录结构

```text
ai_recognition/
├── 01_download_dataset.py          # 下载 Oxford-IIIT Pet 数据
├── 02_prepare_data.py              # 构建原版 37 类数据集
├── 03_train.py                     # 品种识别训练脚本，支持 dataset / dataset_extended
├── 04_export_onnx.py               # 导出品种识别 ONNX
├── 05_test_server.py               # 旧版接口测试脚本
├── 06_download_stanford_dogs.py    # 下载 Stanford Dogs 数据
├── 07_build_extended_dataset.py    # 构建 140 类扩展数据集
├── inference_server.py             # 统一 FastAPI 服务
├── petface/                        # PetFace 2.0 模型加载、推理、评估相关代码
├── scripts/
│   ├── smoke_test_ai_service.py    # 统一服务冒烟测试
│   └── evaluate_petface2_offline.py# PetFace 离线评估脚本
├── models/                         # 1.0 模型与 PetFace 2.0 模型，需单独传输
├── models_extended/                # 1.1 增强版模型，需单独传输
├── dataset/                        # 原版数据集，训练用，不提交 Git
└── dataset_extended/               # 扩展版数据集，训练用，不提交 Git
```

## 当前模型版本

### 品种识别 1.0

- 数据集：Oxford-IIIT Pet Dataset
- 类别数：37
- 任务：猫狗品种分类
- 部署文件：

```text
models/pet_classifier.onnx
models/pet_classifier.onnx.data
models/class_meta.json
```

### 品种识别 1.1 增强版

- 数据集：Oxford-IIIT Pet + Stanford Dogs + Kaggle CatBreedsRefined-7k
- 类别数：140
- 猫：20 类
- 狗：120 类
- 训练规模：train 27550 张，val 3401 张，test 3544 张
- 部署文件：

```text
models_extended/pet_classifier.onnx
models_extended/pet_classifier.onnx.data
models_extended/class_meta.json
```

### PetFace-ID 2.0

- 数据集：PetFace
- 任务：宠物个体识别，而不是品种分类
- 能力：
  - 为宠物档案生成 embedding
  - 判断两张图是否可能是同一只宠物
  - 在系统已建档宠物中检索相似个体
- 部署文件：

```text
models/petface/petface_id_best.pth
models/petface/petface_meta.json
```

## 运行环境

建议使用 conda 环境：

```bash
conda create -n pet python=3.11
conda activate pet
pip install -r requirements.txt
```

Apple Silicon 本地运行时建议：

```bash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## 启动服务

默认启动时加载 `models/` 下的 1.0 品种识别模型：

```bash
cd ai_recognition
conda activate pet
export KMP_DUPLICATE_LIB_OK=TRUE
python inference_server.py
```

推荐演示/部署时加载 1.1 增强版模型：

```bash
cd ai_recognition
conda activate pet
export KMP_DUPLICATE_LIB_OK=TRUE
PET_BREED_MODEL_DIR=models_extended python inference_server.py
```

服务地址：

```text
http://localhost:8000
```

健康检查：

```bash
curl http://localhost:8000/health
```

期望：

```text
breed.loaded = true
petface.loaded = true
petface.embedding_dim = 512
```

## API 接口

### 品种识别

```text
POST /api/recognize
form-data: file
```

返回宠物类型、Top1 品种、中文品种名和 Top5 候选。

### PetFace 特征提取

```text
POST /api/petface/embed
form-data: file
```

返回 512 维归一化 embedding。

### PetFace 同宠验证

```text
POST /api/petface/verify
form-data: file1, file2
```

返回两张图的余弦相似度、是否同宠、置信等级。

## 训练与导出

### 训练 1.1 增强版品种识别

```bash
cd ai_recognition
python 03_train.py --data-dir dataset_extended --model-dir models_extended
```

训练完成后导出 ONNX：

```bash
python 04_export_onnx.py --model-dir models_extended
```

### 构建扩展数据集

```bash
cd ai_recognition
python 06_download_stanford_dogs.py
mkdir -p data_raw/kaggle_catbreedsrefined
kaggle datasets download -d doctrinek/catbreedsrefined-7k -p data_raw/kaggle_catbreedsrefined --unzip
python 07_build_extended_dataset.py --force
```

## 测试

### 统一服务冒烟测试

```bash
cd ai_recognition
python scripts/smoke_test_ai_service.py --image "data_raw/images/Abyssinian_1.jpg"
```

### PetFace 2.0 离线评估

示例：

```bash
python scripts/evaluate_petface2_offline.py \
  --petface-root "/Volumes/ORGOS - Data/PetFaceWorkspace/ai_petface/data/PetFace" \
  --animals dog cat \
  --tasks closed_loop \
  --checkpoint models/petface/petface_id_best.pth \
  --device mps \
  --batch-size 16 \
  --max-closed-loop-identities 500 \
  --out reports/petface2_closed_loop_quick.json
```

## 与系统集成

当前 Python 服务已经被 Spring Boot 后端接入：

- 发布领养/寻宠时，后端可调用 `/api/recognize` 辅助填写宠物种类和品种。
- 宠物建档后，后端调用 `/api/petface/embed` 生成个体特征并写入数据库。
- AI 寻宠时，后端将上传图片转为 embedding，与数据库中所有已建档宠物进行相似检索。

完整部署资产说明见后端仓库：

```text
backEnd/docs/部署资产同步说明_20260519.md
```

当前完整部署资产包：

```text
deploy_packages/pets_deploy_assets_20260519.zip
```

## Git 注意事项

以下内容体积较大或包含运行时资产，不提交 Git：

```text
models/
models_extended/
model_backups/
data_raw/
dataset/
dataset_extended/
reports/
*.log
```

