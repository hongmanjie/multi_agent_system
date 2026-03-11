# SAM3 模型集成说明

## 概述

SAM3 (Segment Anything Model 3) 已成功集成到 MaaS 平台中，支持通过文本提示进行开放词汇目标检测和分割。

## 文件结构

```
xt_maas/models/cv/object_detection/sam3/
├── __init__.py          # 模块导出文件
└── sam3_model.py        # SAM3 模型实现
```

## 配置说明

### model_server.json 配置

```json
{
  "model_id": "sam3",
  "model_type": "cv",
  "module": "xt_maas.models.cv.object_detection.sam3",
  "class": "SAM3Model",
  "enabled": true,
  "model_config": {
    "model_path": "/data_ssd/maas_models/weight/Sam3/sam3.pt",
    "device": "cuda:2",
    "enable_cuda": true,
    "box_threshold": 0.5,
    "text_threshold": 0.25
  }
}
```

### 配置参数说明

- `model_path`: SAM3 模型权重文件路径
- `device`: GPU 设备编号 (如 "cuda:2")
- `enable_cuda`: 是否启用 CUDA 加速
- `box_threshold`: 检测框置信度阈值 (默认 0.5)
- `text_threshold`: 文本匹配置信度阈值 (默认 0.25)

## API 使用

### 启动服务

```bash
# 启动 MaaS 服务
python api_server.py --config_path configs/model_server.json --host 0.0.0.0 --port 8010
```

### API 端点

服务启动后，SAM3 模型的 API 端点为：
```
POST /cv/sam3/predict
```

### 请求格式

```json
{
  "request_id": "unique_request_id",
  "image": "<base64_encoded_image>",
  "categories": ["airplane", "person", "car"],
  "score_threshold": 0.5,
  "roi": []
}
```

#### 请求参数说明

- `request_id`: 唯一请求标识符
- `image`: Base64 编码的图像数据
- `categories`: **必需**，要检测的目标类别列表
- `score_threshold`: 置信度阈值 (可选，默认 0.5)
- `roi`: 感兴趣区域 (可选)，用于限定检测区域

### 响应格式

```json
{
  "model_id": "sam3",
  "request_id": "unique_request_id",
  "image_width": 1920,
  "image_height": 1080,
  "prediction": {
    "count": 2,
    "detections": [
      {
        "bbox": [100, 200, 300, 400],
        "class": "airplane",
        "conf": 0.95,
        "mask": [...]
      }
    ],
    "success": true,
    "processing_time": 1.234,
    "timestamp": "2024-01-01T12:00:00"
  }
}
```

## cURL 示例

```bash
curl -X POST "http://localhost:8010/cv/sam3/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "request_id": "test_sam3_001",
    "image": "<base64_image_data>",
    "categories": ["airplane", "person"],
    "score_threshold": 0.5
  }'
```

## Python 客户端示例

```python
import requests
import base64

# 读取并编码图像
with open("test_image.jpg", "rb") as f:
    image_data = base64.b64encode(f.read()).decode('utf-8')

# 构建请求
url = "http://localhost:8010/cv/sam3/predict"
data = {
    "request_id": "test_001",
    "image": image_data,
    "categories": ["airplane", "person", "car"],
    "score_threshold": 0.5
}

# 发送请求
response = requests.post(url, json=data)
result = response.json()

# 处理结果
print(f"检测到 {result['prediction']['count']} 个目标")
for detection in result['prediction']['detections']:
    print(f"类别：{detection['class']}, 置信度：{detection['conf']:.2f}")
    print(f"边界框：{detection['bbox']}")
```

## 测试

### 运行单元测试

```bash
# 运行 SAM3 模型测试
python tests/test_sam3.py
```

### 测试脚本说明

测试脚本会：
1. 加载 SAM3 模型
2. 使用测试图像进行推理
3. 验证模型输出格式
4. 打印检测结果

## 特性

- ✅ **文本提示检测**: 支持通过文本提示检测任意类别
- ✅ **多类别检测**: 支持同时检测多个类别
- ✅ **实例分割**: 返回目标掩码信息
- ✅ **ROI 支持**: 支持指定感兴趣区域
- ✅ **置信度过滤**: 支持阈值过滤低置信度检测
- ✅ **异步处理**: 基于 FastAPI 的异步请求处理

## 注意事项

1. **类别参数**: `categories` 参数是必需的，必须提供至少一个检测类别
2. **GPU 内存**: SAM3 模型较大，确保有足够的 GPU 内存 (建议>=8GB)
3. **推理速度**: 每个类别需要单独推理，类别数量会影响总推理时间
4. **模型路径**: 确保配置中的模型路径正确且文件存在

## 故障排除

### 模型加载失败

```
错误：模型文件不存在：/path/to/sam3.pt
```
**解决方案**: 检查模型路径配置是否正确，文件是否存在

### CUDA 内存不足

```
RuntimeError: CUDA out of memory
```
**解决方案**: 
- 减小图像尺寸
- 使用更小的 batch size
- 更换到内存更大的 GPU

### 推理结果为空

可能原因：
- 置信度阈值设置过高
- 图像中不存在指定类别的目标
- 类别名称描述不准确

**解决方案**:
- 降低 `box_threshold` 参数
- 尝试不同的类别描述

## 参考资料

- SAM3 原始论文：[链接]
- SAM3 GitHub 仓库：[链接]
- MaaS 平台文档：README.md
