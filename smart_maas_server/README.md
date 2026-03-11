# Smart MaaS (Model-as-a-Service) 平台

Smart MaaS 是一个高效、灵活的模型即服务系统，集成了多种主流AI模型，包括计算机视觉、多模态和自然语言处理模型，旨在提高模型复用性并简化模型推理流程。

## 🌟 特性

- **多模型类型支持**：支持计算机视觉(CV)、多模态(Multi-modal)和自然语言处理(NLP)模型
- **动态模型加载**：通过配置文件动态启用/禁用模型服务
- **统一API接口**：提供标准化的REST API接口，便于集成
- **多实例部署**：同一算法类型可以部署多个实例
- **灵活配置**：支持设备分配、阈值调节等个性化配置
- **异步处理**：基于FastAPI的异步请求处理，提升性能

## 📁 目录结构

```
├── app/                    # 面向上层业务的接口，模型服务的调度
│   ├── app.py             # 主应用入口
│   └── vl_app.py          # 视觉语言应用
├── configs/               # 配置文件
│   ├── download_hf_model.py # 下载HuggingFace模型脚本
│   ├── model_server.json  # 模型服务配置
│   ├── server.json        # 服务器配置
│   └── v3_tasks_config_railway.json # 铁路任务配置示例
├── tests/                 # 测试文件
│   ├── maas_examples/     # MaaS示例
│   ├── test_client.py     # 客户端测试
│   ├── test_face_api.py   # 人脸API测试
│   ├── test_grounding_dino.py # Grounding DINO测试
│   ├── test_minicpmv_video.py # MiniCPMV视频测试
│   └── ...
├── xt_maas/               # 核心模型平台
│   ├── models/            # 模型实现代码
│   │   ├── base/          # 基础模型类
│   │   ├── cv/            # 计算机视觉模型
│   │   ├── multi_modal/   # 多模态模型
│   │   └── nlp/           # 自然语言处理模型
│   ├── server/            # 模型服务代码
│   │   ├── api_server.py  # API服务器主入口
│   │   ├── api_server_simp.py # 简化版API服务器
│   │   └── model_handler.py # 模型处理器
│   └── utils/             # 工具函数
├── api_server.py          # API服务器入口
├── run.sh                 # 运行脚本
└── README.md              # 项目说明文档
```

## 🧠 支持的模型类型

### 计算机视觉 (CV) 模型

- **对象检测**
  - Rex-Omni: 统一视觉基础模型
  - Grounding DINO: 开放词汇目标检测
  - YOLO系列: YOLOv12人体检测、头盔检测、姿态估计
  - DAMO-YOLO: 工业级目标检测

- **人脸相关**
  - RetinaFace: 人脸检测
  - ArcFace: 人脸识别
  - FER: 面部表情识别

### 多模态模型

- **视觉语言模型**
  - MiniCPM-V 4.5: 高效视觉语言模型
  - Qwen2.5-VL: 通义千问视觉语言模型
  - Qwen3-VL: 通义千问三代视觉语言模型
  - DFN5B-CLIP: 大规模对比语言图像预训练模型

### 自然语言处理 (NLP) 模型

- **大语言模型**
  - Qwen2.5: 通义千问2.5
  - Qwen3: 通义千问3
  - Qwen3:7B: 通义千问3-7B版本，部署在cuda:2
  - vLLM优化版本: 基于vLLM框架优化的大语言模型

## ⚙️ 快速开始

### 环境要求

- Python >= 3.8
- CUDA兼容GPU (推荐)
- 适当的模型权重文件

### 安装步骤

1. **克隆仓库**

```bash
git clone <repository-url>
cd smart_maas_server
```

2. **安装依赖**

```bash
pip install -e .
```

3. **配置模型服务**

编辑 `configs/model_server.json` 文件，将需要启用的模型 `enabled` 字段设为 `true`：

```json
{
  "model_id": "minicpmv_4_5",
  "model_type": "multi_modal",
  "enabled": true,
  "model_config": {
    "model_path": "/path/to/model",
    "device_map": "auto"
  }
}
```

4. **启动模型服务**

```bash
# 方式一：直接运行
python -m xt_maas.server.api_server

# 方式二：通过Python脚本
python xt_maas/server/api_server.py --config_path configs/model_server.json --host 0.0.0.0 --port 8010

# 方式三：使用启动脚本
bash run.sh
```

成功启动后可以看到配置好的模型路由列表。路由命名规则为：
`/{model_type}/{model_id}/predict`

### API 使用示例

#### 计算机视觉模型请求

```bash
curl -X POST "http://localhost:8010/cv/yolo_person/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "image": "<base64_encoded_image>",
    "score_threshold": 0.5,
    "nms_threshold": 0.4
  }'
```

#### 多模态模型请求

```bash
curl -X POST "http://localhost:8010/multi_modal/minicpmv_4_5/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "image": "<base64_encoded_image>",
    "text": "请描述这张图片",
    "temperature": 0.7
  }'
```

#### NLP模型请求

```bash
curl -X POST "http://localhost:8010/nlp/qwen2_5_llm/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "prompt": "请解释量子计算的基本概念",
    "temperature": 0.8,
    "max_new_tokens": 512
  }'

# Qwen3-7B模型请求示例
curl -X POST "http://localhost:8010/nlp/qwen3_7b/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_001",
    "prompt": "请解释量子计算的基本概念",
    "temperature": 0.8,
    "max_new_tokens": 512
  }'
```

## 🔧 配置说明

### 模型服务配置 (configs/model_server.json)

此文件定义了系统中可用的所有模型及其配置：

- `model_id`: 模型唯一标识符
- `model_type`: 模型类型 (cv, multi_modal, nlp)
- `module`: 模型模块路径
- `class`: 模型类名
- `enabled`: 是否启用此模型
- `model_config`: 模型特定配置

### 服务器配置 (configs/server.json)

定义了特定任务的配置，如标签、提示词、响应消息等。

## 🧪 测试指南

### 单元测试

对单独的算法模块进行单元测试：

```bash
# 安装包以确保模块正确调用
pip install -e .

# 运行各种模型的测试
python tests/test_face_detection.py
python tests/test_arcface_recognition.py
python tests/test_face_expression_recognition.py
python tests/test_yolo.py
python tests/test_gounding_dino.py
python tests/test_qwen2_5_vl.py
python tests/test_clip.py
```

### 客户端测试

```bash
# 测试启用的模型服务
python tests/test_client.py
```

### 应用服务测试

```bash
# 启动app服务
python app/app.py

# 测试app服务
python tests/test_app.py
```

## 📊 API 文档

### 通用请求格式

所有API请求都遵循以下格式：

```json
{
  "request_id": "唯一请求标识符",
  "image": "base64编码的图像数据(可选)",
  "text": "文本输入(可选)",
  "prompt": "提示词(可选)",
  "...": "其他模型特定参数"
}
```

### 通用响应格式

```json
{
  "model_id": "模型ID",
  "request_id": "请求ID",
  "image_width": "图像宽度(如果有图像输入)",
  "image_height": "图像高度(如果有图像输入)",
  "prediction": "预测结果"
}
```

### 不同模型类型的具体参数

#### CV模型参数
- `image`: base64编码的图像
- `categories`: 检测类别列表
- `nms_threshold`: 非极大值抑制阈值
- `score_threshold`: 置信度阈值
- `roi`: 感兴趣区域

#### 多模态模型参数
- `image`: base64编码的图像(可选)
- `text`: 文本提示词(可选)
- `temperature`: 温度参数
- `top_p`: Top-P采样参数

#### NLP模型参数
- `prompt`: 输入提示词
- `temperature`: 温度参数
- `top_p`: Top-P采样参数
- `top_k`: Top-K采样参数
- `repetition_penalty`: 重复惩罚系数
- `max_new_tokens`: 最大生成token数

