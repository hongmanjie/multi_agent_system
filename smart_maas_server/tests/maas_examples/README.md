# MaaS 智能视觉检测编排器

## 项目简介

`main.py` 是一个基于 MaaS (Model as a Service) 架构的智能视觉检测编排器，集成了 YOLO 目标检测和视觉语言大模型 (VL) 进行智能场景分析。主要用于跌倒检测和打架斗殴检测等安全监控场景。

## 核心功能

### 1. 多模态AI任务编排
- **YOLO目标检测**: 检测图片中的人物数量
- **视觉语言模型分析**: 使用大模型进行场景理解和判断
- **智能决策**: 结合检测结果和模型分析进行最终判断

### 2. 支持的任务类型
- **跌倒检测** (`FALL_DETECTION`): 检测画面中是否有人跌倒
- **打架检测** (`FIGHT_DETECTION`): 检测画面中是否有打架斗殴行为

### 3. 支持的模型
- **qwen2_5_vl**: 通义千问视觉语言模型
- **minicpmv_4_5**: MiniCPM 视觉语言模型

## 系统架构

```
输入图片 → YOLO检测 → 人物数量判断 → VL模型分析 → 结果输出
```

### 工作流程
1. **图片预处理**: 将输入图片转换为Base64格式
2. **目标检测**: 使用YOLO模型检测图片中的人物
3. **数量验证**: 检查检测到的人物数量是否满足任务要求
4. **场景分析**: 使用VL大模型对场景进行深度分析
5. **结果判定**: 根据VL模型的回答进行最终判断

## 安装依赖

```bash
pip install requests pydantic
```

## 使用方法

### 基本使用

```python
from main import MaaSOrchestrator, TaskType

# 初始化编排器
orchestrator = MaaSOrchestrator("http://your-server:8000")

# 执行跌倒检测
result = orchestrator.execute_task(
    image_path="test_image.jpg",
    task_type=TaskType.FALL_DETECTION,
    model_id="qwen2_5_vl"
)

print(f"检测结果: {result}")
```

### 自定义配置

```python
from main import TaskConfig, TaskType

# 创建自定义任务配置
custom_config = TaskConfig(
    task_type=TaskType.FIGHT_DETECTION,
    min_persons=3,  # 最少需要3个人
    vl_question="请判断画面中是否有暴力行为...",
    yolo_config={
        "nms_threshold": 0.5,
        "score_threshold": 0.6
    }
)

# 使用自定义配置执行任务
result = orchestrator.execute_task(
    image_path="test_image.jpg",
    task_type=TaskType.FIGHT_DETECTION,
    custom_config=custom_config
)
```

## 配置说明

### TaskConfig 参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `task_type` | TaskType | - | 任务类型 (FALL_DETECTION/FIGHT_DETECTION) |
| `min_persons` | int | 1/2 | 最少需要检测到的人物数量 |
| `vl_question` | str | - | 发送给VL模型的问题 |
| `yolo_config` | dict | 见下表 | YOLO检测配置 |
| `vl_config` | dict | `{"temperature": 0.1}` | VL模型配置 |
| `positive_keywords` | list | `["是", "yes", "true"]` | 肯定回答关键词 |
| `negative_keywords` | list | `["否", "no", "false"]` | 否定回答关键词 |

### YOLO配置参数

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `nms_threshold` | float | 0.4 | 非极大值抑制阈值 |
| `score_threshold` | float | 0.5 | 置信度阈值 |
| `roi` | list | [] | 感兴趣区域 |

## 返回结果格式

```python
{
    "success": True,           # 是否成功执行
    "result": True,            # 最终判断结果
    "reason": "VL模型判断",     # 判断原因
    "person_count": 2,         # 检测到的人物数量
    "vl_answer": "是",         # VL模型的原始回答
    "yolo_details": {...},     # YOLO检测详细结果
    "vl_details": {...}        # VL模型详细结果
}
```

## 错误处理

系统包含完善的错误处理机制：

- **图片读取失败**: 自动捕获并记录错误
- **网络请求超时**: 设置30-60秒超时时间
- **模型服务不可用**: 返回失败状态和错误信息
- **结果解析失败**: 提供降级处理方案

## 日志系统

系统使用Python标准logging模块，支持以下日志级别：
- `INFO`: 正常执行信息
- `WARNING`: 警告信息（如VL回答不明确）
- `ERROR`: 错误信息

## 扩展开发

### 添加新的任务类型

1. 在 `TaskType` 枚举中添加新类型
2. 在 `_load_default_configs()` 方法中添加默认配置
3. 根据需要调整检测逻辑

### 添加新的模型

在 `execute_task()` 方法中通过 `model_id` 参数指定新模型，系统会自动调用对应的API端点。

## 注意事项

1. **服务器地址**: 确保MaaS服务器地址正确且可访问
2. **图片格式**: 支持常见图片格式 (jpg, png, bmp等)
3. **网络环境**: 需要稳定的网络连接访问模型服务
4. **资源消耗**: VL模型推理可能消耗较多计算资源

## 示例运行

```bash
python main.py
```

运行后会执行以下测试：
- 使用qwen2_5_vl模型进行跌倒检测
- 使用minicpmv_4_5模型进行跌倒检测  
- 使用qwen2_5_vl模型进行打架检测
- 使用minicpmv_4_5模型进行打架检测

## 技术特点

- **模块化设计**: 清晰的类结构和职责分离
- **类型安全**: 使用Pydantic进行数据验证
- **可扩展性**: 易于添加新的任务类型和模型
- **容错性**: 完善的异常处理和降级机制
- **可配置性**: 灵活的参数配置系统

## 版本信息

- Python版本: 3.7+
- 主要依赖: requests, pydantic
- 架构模式: MaaS (Model as a Service)
