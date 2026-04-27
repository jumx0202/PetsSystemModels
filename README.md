# AI 品种识别模块 (AI Recognition)

本项目是宠物管理系统（毕业设计）的 AI 识别子模块，主要负责通过计算机视觉技术对宠物（狗、猫）的品种进行自动识别。该模块以独立的服务形式运行，通过 HTTP 接口向 Spring Boot 后端提供推理结果。

## 目录结构与功能

整个 AI 模型的训练与推理流程被拆分为多个步骤，分别对应不同的脚本文件：

- **01_download_dataset.py**：用于下载并解压宠物品种数据集。
- **02_prepare_data.py**：数据集预处理，包括图像清洗、尺寸调整和划分训练/验证集。
- **03_train.py**：使用 PyTorch 训练图像分类模型，基于预训练模型进行微调（Fine-tuning），保存 `.pth` 权重文件。
- **04_export_onnx.py**：将训练好的 PyTorch 模型导出为 ONNX 格式（`pet_classifier.onnx`），以提升后续推理的性能与部署便利性。
- **05_test_server.py**：用于测试推理服务接口的脚本。
- **inference_server.py**：核心的推理服务端。基于 FastAPI 和 ONNX Runtime 构建，提供 HTTP API（`POST /api/recognize`）供外部调用。
- **demo.py**：可能包含一些推理演示代码或界面。
- **models/**：存放训练出的 PyTorch 权重（`.pth`）、ONNX 模型文件（`.onnx`）以及类别映射配置（`class_meta.json`）。
- **AI品种识别技术总结.md** & **行为分析方案.md**：技术方案、算法原理和实现细节的总结文档，适合用于论文撰写参考。

## 当前进度

目前 AI 识别模块已经完成了从数据准备到模型部署的 **全流程闭环**：
1. **数据准备与训练 (已完成)**：数据集处理、模型训练和验证均已完成，效果最好的权重已经保存在 `models/` 目录下。
2. **模型转换 (已完成)**：模型已成功导出为 `pet_classifier.onnx`，为轻量化、高性能推理做好了准备。
3. **推理服务 (已完成)**：`inference_server.py` 已开发完毕，对外暴露 8000 端口，可成功接收图片并返回 Top-N 品种及置信度。
4. **前后端联调 (进行中/已走通)**：Spring Boot 后端（`AiRecognizeController`）已被配置为转发前端的请求至此模块的推理服务，目前已经打通了从 Vue 前端 -> Java 后端 -> Python 推理接口的数据链路。

## 如何运行服务

若要在本地启动 AI 识别服务，请确保已安装 `requirements.txt` 中的依赖（包括 `fastapi`, `uvicorn`, `onnxruntime`, `Pillow`, `numpy` 等），然后运行：

```bash
python inference_server.py
```

服务将在 `http://localhost:8000` 启动，主接口为 `POST /api/recognize`。
