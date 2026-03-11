import cv2
import base64
import httpx

from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
import json
from pathlib import Path
import uuid
import tempfile
import os
from datetime import datetime
from funasr import AutoModel

app = FastAPI()

# 加载JSON配置文件
# CONFIG_FILE = "./configs/server.json"
# with open(CONFIG_FILE, "r", encoding="utf-8") as f:
#     CONFIG = json.load(f)

SERVER_URL = "http://localhost:8000"

API_ENDPOINTS = {
    "rex_endpoint": "/cv/rex_omni/predict",
    "person_endpoint": "/cv/yolo_person/predict",
    "helmet_endpoint": "/cv/yolo_helmet/predict",
    "keypoint_endpoint": "/cv/yolo_pose/predict"
}

class CvRequest(BaseModel):
    request_id: str 
    image: str  # base64编码的图片
    score: float = 0.3
    iou: float = 0.5
    fps: float = 1.0
    duration: float = 1.0
    roi: List[List[float]] = [] # 检测区域，每个区域是一个闭合坐标点序列

class DetectionResponse(BaseModel):
    count: int
    detections: List[Dict[str, Any]]
    height: int
    width: int
    warning: bool = False
    inference_time: float = 0.0


cv_api_router = APIRouter() # yolo

@cv_api_router.post("/v1/detect_person", response_model=DetectionResponse)
async def detect_person(request: CvRequest):

    url = f"{SERVER_URL}{API_ENDPOINTS['person_endpoint']}"
    data = {
        "request_id": request.request_id,
        "image": request.image,
        "score_threshold": request.score,
        "nms_threshold": request.iou,
        "roi": request.roi
        }
    response = requests.post(url=url, json=data)
    response.raise_for_status()
    api_result = response.json()
    print("==yolo result==:")
    print(api_result)
    yolo_results = api_result['prediction']
    image_width = api_result['image_width']
    image_height = api_result['image_height']
    
    return DetectionResponse(
        count=yolo_results['count'],
        detections=yolo_results['detections'],
        height=image_height,
        width=image_width,
        warning=yolo_results['count'] > 0,
        inference_time=yolo_results['processing_time']
    )

@cv_api_router.post("/v1/detect_helmet", response_model=DetectionResponse)
async def detect_helmet(request: CvRequest):

    url = f"{SERVER_URL}{API_ENDPOINTS['helmet_endpoint']}"
    data = {
        "request_id": request.request_id,
        "image": request.image,
        "score_threshold": request.score,
        "nms_threshold": request.iou,
        "roi": request.roi
        }
    response = requests.post(url=url, json=data)
    response.raise_for_status()
    api_result = response.json()
    print("==yolo result==:")
    print(api_result)
    yolo_results = api_result['prediction']
    image_width = api_result['image_width']
    image_height = api_result['image_height']
    
    return DetectionResponse(
        count=yolo_results['count'],
        detections=yolo_results['detections'],
        height=image_height,
        width=image_width,
        warning=yolo_results['count'] > 0,
        inference_time=yolo_results['processing_time']
    )

@cv_api_router.post("/v1/detect_keypoint", response_model=DetectionResponse)
async def detect_helmet(request: CvRequest):

    url = f"{SERVER_URL}{API_ENDPOINTS['keypoint_endpoint']}"
    data = {
        "request_id": request.request_id,
        "image": request.image,
        "score_threshold": request.score,
        "nms_threshold": request.iou,
        "roi": request.roi
        }
    response = requests.post(url=url, json=data)
    response.raise_for_status()
    api_result = response.json()
    print("==yolo result==:")
    print(api_result)
    yolo_results = api_result['prediction']
    image_width = api_result['image_width']
    image_height = api_result['image_height']
    
    return DetectionResponse(
        count=yolo_results['count'],
        detections=yolo_results['detections'],
        height=image_height,
        width=image_width,
        warning=yolo_results['count'] > 0,
        inference_time=yolo_results['processing_time']
    )
# @owd_api_router.post("/v2/{task_name}", response_model=DetectionResponse)
# async def owd_analysis_endpoint(request: ROIRequest, task_name: str = None):
#     tasks = CONFIG["tasks"]
#     if task_name not in tasks:
#         raise HTTPException(status_code=404, detail=f"任务名 '{task_name}' 未在服务任务配置中找到")
    
#     target_label = tasks[task_name]['label']
#     message_true = tasks[task_name]['message_true']
#     message_false = tasks[task_name]['message_false']
#     url = f"{SERVER_URL}{API_ENDPOINTS['owd_endpoint']}"
#     data = {
#         "request_id": str(uuid.uuid4()),
#         "image": request.image
#         }
#     response = requests.post(url=url, json=data)
#     response.raise_for_status()
#     api_result = response.json()
#     print("==dino result==:")
#     print(api_result)
#     # 
#     dino_results = api_result['prediction']
#     has_target = any(label == target_label for label in dino_results['labels'])

#     details = [
#         {'position': dino_results['boxes'][i], 'confidence': dino_results['scores'][i], 'label': target_label}
#         for i, label in enumerate(dino_results["labels"])
#         if label.lower() == target_label
#     ]
    
#     return DetectionResponse(
#         has_target=has_target,
#         details=details,
#         message=message_true if has_target else message_false
#     )

# @mm_api_router.post("/v3/{task_name}", response_model=DetectionResponse)
# async def mm_analysis_endpoint(request: DescriptionRequest, task_name: str = None):
#     tasks = CONFIG["tasks"]
#     if task_name not in tasks:
#         raise HTTPException(status_code=404, detail=f"任务名 '{task_name}' 未在服务任务配置中找到")

#     if request.text:
#         # request传入的prompt
#         target_prompt = request.text
#     else:
#         # 否则使用config中配置的默认prompt
#         target_prompt =  tasks[task_name]['prompt']
#     print(target_prompt)
#     message_true = tasks[task_name]['message_true']
#     message_false = tasks[task_name]['message_false']

#     url = f"{SERVER_URL}{API_ENDPOINTS['mm_endpoint']}"
#     data = {
#         "request_id": str(uuid.uuid4()),
#         "image": request.image,
#         "text": target_prompt
#         }
#     response = requests.post(url=url, json=data)
#     response.raise_for_status()
#     api_result = response.json()
#     print("==mm result==:")
#     print(api_result)

#     has_target = api_result['prediction'].strip().lower() == '是'
#     return DetectionResponse(
#         has_target=has_target,
#         details=[],
#         message=message_true if has_target else message_false
#     )
    

app.include_router(cv_api_router)

# 音频转写路由
audio_api_router = APIRouter()

@audio_api_router.post("/transcribe", response_model=Dict[str, Any])
async def transcribe_audio(
    audio_file: UploadFile = File(..., description="音频文件 (WAV/MP3)"),
    sample_rate: int = 16000
):
    """
    语音转文字接口
    - 支持格式: WAV, MP3
    - 推荐采样率: 16kHz
    - 建议音频长度: >40秒以获得更好的说话人区分效果
    """
    # 检查文件类型
    allowed_types = ['audio/wav', 'audio/mpeg', 'audio/mp3', 'application/octet-stream']
    if audio_file.content_type not in allowed_types:
        raise HTTPException(status_code=400, detail="不支持的音频格式")
    
    # 读取音频数据
    try:
        audio_data = await audio_file.read()
        
        # 获取文件扩展名
        file_extension = audio_file.filename.split('.')[-1].lower() if audio_file.filename else 'wav'
        if file_extension not in ['wav', 'mp3']:
            file_extension = 'wav'
        
        # 编码音频数据为base64
        import base64
        audio_data_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # 调用smart_maas_server的音频模型
        url = "http://localhost:8010/audio/funasr_asr/predict"
        data = {
            'request_id': str(uuid.uuid4()),
            'audio_data': audio_data_base64,
            'audio_format': file_extension
        }
        
        response = requests.post(url=url, json=data, timeout=60)
        response.raise_for_status()
        api_result = response.json()
        
        if 'prediction' in api_result:
            return api_result['prediction']
        else:
            raise HTTPException(status_code=500, detail="服务返回格式错误")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"处理失败: {str(e)}")

app.include_router(audio_api_router)

# === 启动入口 ===
if __name__ == "__main__":
    import uvicorn
    # 🔍 打印所有已注册的路由路径（调试用）
    print("\n🔍 服务启动后，已注册的路由列表：")
    for route in app.routes:
        print(f"  - {route.path}")
    uvicorn.run(app, host="0.0.0.0", port=7777)




