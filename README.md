# AI 统一识别模块 (AI Recognition)

本项目是宠物管理系统（毕业设计）的 AI 识别子模块。当前模块已经由单一“品种识别 1.0”升级为统一 FastAPI 模型服务，同时承载：

- **品种识别 1.0**：识别猫狗品种，接口为 `POST /api/recognize`。
- **PetFace-ID 2.0 个体识别**：提取宠物个体特征、判断两张图片是否为同一只宠物，接口为 `POST /api/petface/embed` 与 `POST /api/petface/verify`。

该模块以独立服务形式运行，通过 HTTP 接口向 Spring Boot 后端提供推理结果。

## 目录结构与功能

整个 AI 模型的训练与推理流程被拆分为多个步骤，分别对应不同的脚本文件：

- **01_download_dataset.py**：用于下载并解压宠物品种数据集。
- **02_prepare_data.py**：数据集预处理，包括图像清洗、尺寸调整和划分训练/验证集。
- **03_train.py**：使用 PyTorch 训练图像分类模型，基于预训练模型进行微调（Fine-tuning），保存 `.pth` 权重文件。
- **04_export_onnx.py**：将训练好的 PyTorch 模型导出为 ONNX 格式（`pet_classifier.onnx`），以提升后续推理的性能与部署便利性。
- **05_test_server.py**：用于测试推理服务接口的脚本。
- **inference_server.py**：统一推理服务端。基于 FastAPI，提供品种识别与 PetFace 个体识别接口。
- **petface/**：PetFace-ID 2.0 推理相关代码。
- **models/**：存放训练出的 PyTorch 权重（`.pth`）、ONNX 模型文件（`.onnx`）以及类别映射配置（`class_meta.json`）。
- **models/petface/**：存放 PetFace-ID 2.0 最佳模型与元信息。
- **AI品种识别技术总结.md** & **行为分析方案.md**：技术方案、算法原理和实现细节的总结文档，适合用于论文撰写参考。

## 当前进度

目前 AI 识别模块已经完成了从数据准备到模型部署的 **全流程闭环**：
1. **数据准备与训练 (已完成)**：数据集处理、模型训练和验证均已完成，效果最好的权重已经保存在 `models/` 目录下。
2. **模型转换 (已完成)**：1.0 模型已成功导出为 `pet_classifier.onnx`，为轻量化、高性能推理做好了准备。
3. **PetFace 2.0 (已完成训练并接入统一服务)**：最佳模型已保存到 `models/petface/petface_id_best.pth`，统一服务可输出 512 维个体特征。
4. **推理服务 (已完成 Python 端合并)**：`inference_server.py` 对外暴露 8000 端口，可同时处理品种识别和个体识别。
5. **前后端联调 (下一阶段)**：后续 Spring Boot 后端将接入 PetFace embedding 保存、同宠验证和相似宠物检索。

## 扩展版品种数据集

原始 1.0 版本使用 Oxford-IIIT Pet Dataset，共 37 个猫狗品种。为了增强毕业设计的模型任务难度与覆盖范围，当前已新增 Stanford Dogs Dataset 与 Kaggle CatBreedsRefined-7k 作为扩展数据源，并构建了扩展版训练集：

- 原始数据集：Oxford-IIIT Pet Dataset，37 类，约 7.4k 张图。
- 新增数据集：Stanford Dogs Dataset，120 个犬种，约 20.6k 张图。
- 新增数据集：Kaggle CatBreedsRefined-7k，20 个猫品种，7000 张图，License 为 CC-BY-SA-4.0。
- 扩展后数据集：`dataset_extended/`，共 140 类，其中猫 20 类、狗 120 类。
- 数据规模：train 27550 张，val 3401 张，test 3544 张。
- 重复/同义品种已合并，例如 `basset` 合并为 `basset_hound`，`leonberg` 合并为 `leonberger`，`soft_coated_wheaten_terrier` 合并为 `wheaten_terrier`。
- 猫品种重复/同义类别已合并，例如 `British Shorthair` 合并为 `British_Shorthair`，`Egyptian Mau` 合并为 `Egyptian_Mau`，`Maine Coon` 合并为 `Maine_Coon`。
- 新增猫品种包括 `American_Bobtail`、`American_Curl`、`American_Shorthair`、`Exotic_Shorthair`、`Manx`、`Norwegian_Forest`、`Scottish_Fold`、`Turkish_Angora`。
- 默认排除了 `dingo`、`dhole`、`african_hunting_dog` 等野生犬科类别，使任务更贴近宠物品种识别。

相关脚本：

- `06_download_stanford_dogs.py`：下载并解压 Stanford Dogs Dataset。
- `07_build_extended_dataset.py`：将 Oxford-IIIT Pet、Stanford Dogs 与 Kaggle CatBreedsRefined-7k 合并为 `dataset_extended/`。

复现扩展数据集：

```bash
cd ai_recognition
python 06_download_stanford_dogs.py
mkdir -p data_raw/kaggle_catbreedsrefined
kaggle datasets download -d doctrinek/catbreedsrefined-7k -p data_raw/kaggle_catbreedsrefined --unzip
python 07_build_extended_dataset.py --force
```

训练扩展版模型：

```bash
cd ai_recognition
python 03_train.py --data-dir dataset_extended --model-dir models_extended
```

导出扩展版 ONNX：

```bash
cd ai_recognition
python 04_export_onnx.py --model-dir models_extended
```

启动扩展版推理服务：

```bash
cd ai_recognition
PET_BREED_MODEL_DIR=models_extended python inference_server.py
```

## PetFace-ID 2.0 模型

PetFace-ID 2.0 当前使用 PetFace 数据集训练，核心任务是个体识别，而不是品种分类。它可以用于：

- 用户登记宠物时，为宠物头像或照片生成 embedding。
- 用户上传走失/发现宠物图片时，与系统中已登记宠物进行相似检索。
- 判断两张图片是否可能属于同一只宠物。

当前模型文件：

```text
models/petface/petface_id_best.pth
models/petface/petface_meta.json
```

为了避免模型丢失，另有本地备份：

```text
model_backups/petface_pet_species_full_continue_20260518/
```

## 如何运行统一服务

若要在本地启动 AI 识别服务，请确保已安装 `requirements.txt` 中的依赖，然后运行：

```bash
cd ai_recognition
export KMP_DUPLICATE_LIB_OK=TRUE
python inference_server.py
```

服务将在 `http://localhost:8000` 启动。

默认会加载旧版 37 类 ONNX：

```text
models/pet_classifier.onnx
```

如果要使用扩展版 140 类模型，需要先等训练完成并导出 ONNX：

```bash
cd ai_recognition
python 04_export_onnx.py --model-dir models_extended
```

然后启动时指定：

```bash
cd ai_recognition
export KMP_DUPLICATE_LIB_OK=TRUE
PET_BREED_MODEL_DIR=models_extended python inference_server.py
```

## 统一服务接口

健康检查：

```bash
curl http://localhost:8000/health
```

品种识别 1.0：

```text
POST /api/recognize
form-data: file
```

PetFace 特征提取 2.0：

```text
POST /api/petface/embed
form-data: file
```

PetFace 同宠验证 2.0：

```text
POST /api/petface/verify
form-data: file1, file2
```

## Smoke Test

启动服务后，可以运行：

```bash
cd ai_recognition
python scripts/smoke_test_ai_service.py --image "data_raw/images/Abyssinian_1.jpg"
```

期望结果：

- `/health` 中 `breed.loaded=true`
- `/health` 中 `petface.loaded=true`
- `petface embed` 的 `embedding_dim=512`
- 同一张图片验证时 `same_pet=true`，`similarity` 接近 `1.0`
