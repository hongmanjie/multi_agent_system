import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Dict, TypeAlias
import numpy as np
import cv2
from PIL import Image

# 定义视频帧数据类型别名 - API接收时为4D列表格式（多帧），内部处理时转为numpy数组
VideoFrameData: TypeAlias = List[List[List[List[int]]]]  # 对应 numpy array of shape (N, H, W, C)，N为帧数

# 创建FastAPI应用实例
app = FastAPI(
    title="Qwen3-VL 视频分析服务",
    description="基于Qwen3-VL模型的视频内容分析API，支持视频路径和视频帧输入",
    version="1.0.0"
)

# 加载模型和处理器
print("正在加载模型...")
model_path = "/data_ssd/modelscope/hub/models/qwen/Qwen3-VL"
processor = AutoProcessor.from_pretrained(model_path)

# 加载视觉-文本模型
model = Qwen3VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto"
)
print("模型加载完成")

def load_video_frames(video_path, max_frames=4):
    """
    加载视频并提取帧
    Args:
        video_path: 视频文件路径
        max_frames: 最大帧数
    Returns:
        视频帧列表
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        # 转换为 RGB 格式
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 转换为 PIL Image
        frame = Image.fromarray(frame)
        frames.append(frame)
    
    cap.release()
    return frames

# 请求模型 - 视频路径输入（兼容demo版本）
class VideoAnalysisRequest(BaseModel):
    video_path: str
    prompt: str
    sample_fps: Optional[float] = 0.2
    max_frames: Optional[int] = 64
    total_pixels: Optional[int] = 16384 * 32 * 32
    min_pixels: Optional[int] = 64 * 32 * 32
    max_new_tokens: Optional[int] = 2048
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.9

# 请求模型 - 视频路径输入（原有版本）
# class VideoPathRequest(BaseModel):
#     video_path: str
#     prompt: Optional[str] = "请描述这视频中发生的事情"
#     max_frames: Optional[int] = 4

# 请求模型 - 视频帧输入
class VideoFramesRequest(BaseModel):
    """
    视频帧分析请求模型
    
    说明：
    - `frame_data` 是一个四层嵌套列表，表示多帧图像数据：
      1. 第一层：帧索引（视频中的第几帧）
      2. 第二层：图像高度（行数）
      3. 第三层：图像宽度（列数）
      4. 第四层：BGR通道值（因为OpenCV读取的是BGR格式）
    - 例如：[[[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [255, 255, 255]]]] 表示一个包含1帧的2x2像素的视频
    
    注意：
    - 如果你的 `frame_data` 是 numpy 数组（OpenCV 返回的格式），需要先转换为列表才能通过 HTTP API 传递
    - 转换方法：`frame_data_list = frame_data.tolist()`
    - 这是因为 HTTP 请求体需要是 JSON 格式，而 numpy 数组不能直接序列化为 JSON
    
    示例：
    - 输入：`[[[[0, 0, 255], [0, 255, 0]], [[255, 0, 0], [255, 255, 255]]]]`
    - 表示一个包含一帧的视频数据
    """
    frame_data: VideoFrameData  # 4D列表表示的视频数据 (N, H, W, C)，N为帧数
    prompt: Optional[str] = "请描述这视频中发生的事情"

# 响应模型（兼容demo版本）
class VideoAnalysisResponse(BaseModel):
    result: str
    video_path: Optional[str] = None
    input_type: Optional[str] = None  # "video_path" 或 "video_frames"
    processing_time: Optional[str] = None

@app.post("/analyze-video", response_model=VideoAnalysisResponse)
async def analyze_video(request: VideoAnalysisRequest):
    """
    分析视频内容并返回结果
    
    Args:
        request: 包含视频路径、提示词和处理参数的请求对象
        
    Returns:
        包含分析结果的响应对象
    """
    try:
        # 验证视频文件是否存在
        if not os.path.exists(request.video_path):
            raise HTTPException(status_code=404, detail=f"视频文件不存在: {request.video_path}")
        
        # 构建对话消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "video": request.video_path,
                        "total_pixels": request.total_pixels,
                        "min_pixels": request.min_pixels,
                        "max_frames": request.max_frames,
                        "sample_fps": request.sample_fps
                    },
                    {
                        "type": "text",
                        "text": request.prompt
                    },
                ]
            },
        ]
        
        # 应用对话模板
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        # 处理视觉信息
        image_inputs, video_inputs = process_vision_info(
            messages
        )
        
        # 准备模型输入
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            return_tensors="pt"
        )
        inputs = inputs.to(model.device)
        
        # 生成输出
        output_ids = model.generate(
            **inputs, 
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            num_beams=1,
            early_stopping=False
        )
        
        # 提取生成的部分（去除输入）
        generated_ids = [
            output_ids[0][len(inputs.input_ids[0]):]
        ]
        
        # 解码为文本
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )
        
        # 返回结果
        return VideoAnalysisResponse(
            result=output_text[0],
            video_path=request.video_path
        )
        
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"错误详情: {error_trace}")
        raise HTTPException(status_code=500, detail=f"处理视频时出错: {str(e)}")


# @app.post("/analyze-video-by-path", response_model=VideoAnalysisResponse)
# async def analyze_video_by_path(request: VideoPathRequest):
#     """
#     通过视频路径分析视频内容并返回结果
    
#     Args:
#         request: 包含视频路径、提示词的请求对象
        
#     Returns:
#         包含分析结果的响应对象
#     """
#     try:
#         # 验证视频文件是否存在
#         if not os.path.exists(request.video_path):
#             raise HTTPException(status_code=404, detail=f"视频文件不存在: {request.video_path}")
        
#         # 加载视频帧
#         video_frames = load_video_frames(request.video_path, request.max_frames)
        
#         if not video_frames:
#             raise HTTPException(status_code=400, detail="无法加载视频帧，请检查视频文件")
        
#         # 构建消息
#         messages = [
#             {
#                 "role": "user",
#                 "content": [
#                     {
#                         "type": "video",
#                         "video": video_frames,
#                     },
#                     {"type": "text", "text": request.prompt},
#                 ],
#             }
#         ]
        
#         # 处理输入
#         inputs = processor.apply_chat_template(
#             messages,
#             tokenize=True,
#             add_generation_prompt=True,
#             return_dict=True,
#             return_tensors="pt"
#         )
        
#         # 移动到设备
#         inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
#         # 生成输出
#         generated_ids = model.generate(**inputs, max_new_tokens=512)
#         generated_ids_trimmed = [
#             out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
#         ]
#         output_text = processor.batch_decode(
#             generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
#         )
        
#         # 返回结果
#         return VideoAnalysisResponse(
#             result=output_text[0],
#             input_type="video_path",
#             video_path=request.video_path
#         )
        
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"处理视频时出错: {str(e)}")

@app.post("/analyze-video-by-frames", response_model=VideoAnalysisResponse)
async def analyze_video_by_frames(request: VideoFramesRequest):
    """
    通过视频帧分析视频内容并返回结果
    
    Args:
        request: 包含视频帧、提示词的请求对象
        
    Returns:
        包含分析结果的响应对象
    """
    try:
        # 检查帧数据是否为空
        if not request.frame_data:
            raise HTTPException(status_code=400, detail="无法处理视频帧，请检查帧数据格式")
        
        # 将列表数据转换为PIL Image列表（模型期望RGB格式的PIL图像）
        # request.frame_data 是一个4D列表 (N, H, W, C)，其中N是帧数
        pil_frames = []
        for frame_list in request.frame_data:
            # 每一帧都是一个3D列表 (H, W, C)
            frame_np = np.array(frame_list, dtype=np.uint8)
            
            # 确保是3通道图像
            if frame_np.ndim != 3 or frame_np.shape[2] != 3:
                raise HTTPException(status_code=400, detail="帧数据格式不正确，应为(H, W, C)格式的3通道图像")
            
            # 从BGR转换为RGB（假设输入是BGR格式，因为OpenCV默认使用BGR）
            if frame_np.shape[2] == 3:  # 确保有3个通道
                rgb_frame = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame_np  # 如果已经是RGB或其他格式，则保持不变
            
            # 转换为PIL Image
            pil_image = Image.fromarray(rgb_frame)
            pil_frames.append(pil_image)
        
        # 构建消息
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "video",
                        "video": pil_frames,  # 模型期望视频帧列表（PIL图像列表）
                    },
                    {"type": "text", "text": request.prompt},
                ],
            }
        ]
        
        # 处理输入
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # 移动到设备
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # 生成输出
        generated_ids = model.generate(**inputs, max_new_tokens=512)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        # 返回结果
        return VideoAnalysisResponse(
            result=output_text[0],
            input_type="video_frames"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理视频帧时出错: {str(e)}")

@app.get("/")
async def root():
    """
    根路径，返回服务状态
    """
    return {"message": "Qwen3-VL 视频分析服务运行中"}

@app.get("/health")
async def health_check():
    """
    健康检查端点
    """
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
