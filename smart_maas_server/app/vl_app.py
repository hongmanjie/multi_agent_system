import base64
import httpx
import json
import uuid
from fastapi import FastAPI, HTTPException, APIRouter, UploadFile, File
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import requests
from pathlib import Path

app = FastAPI()

# 加载JSON配置文件
CONFIG_FILE = "./configs/server.json"
with open(CONFIG_FILE, "r", encoding="utf-8") as f:
    CONFIG = json.load(f)

SERVER_URL = "http://localhost:8000"

API_ENDPOINTS = {
    "mm_endpoint": "/multi_modal/qwen2_5_vl/predict"
}

class DescriptionRequest(BaseModel):
    image: str  # base64编码的图片
    text: Optional[str] = None  # 可选的自定义提示词

class DetectionResponse(BaseModel):
    has_target: bool  # 是否检测到目标
    details: List[dict]  # 检测详情
    message: str  # 附加信息
    verified: Optional[bool] = None  # v3接口新增，表示是否复核通过

# /v3/has_person - 只保留多模态接口
mm_api_router = APIRouter() # multi_modal

@mm_api_router.post("/v3/{task_name}", response_model=DetectionResponse)
async def mm_analysis_endpoint(request: DescriptionRequest, task_name: str = None):
    tasks = CONFIG["tasks"]
    if task_name not in tasks:
        raise HTTPException(status_code=404, detail=f"任务名 '{task_name}' 未在服务任务配置中找到")

    if request.text:
        # request传入的prompt
        target_prompt = request.text
    else:
        # 否则使用config中配置的默认prompt
        target_prompt =  tasks[task_name]['prompt']
    print(f"使用的提示词: {target_prompt}")
    message_true = tasks[task_name]['message_true']
    message_false = tasks[task_name]['message_false']

    url = f"{SERVER_URL}{API_ENDPOINTS['mm_endpoint']}"
    data = {
        "request_id": str(uuid.uuid4()),
        "image": request.image,
        "text": target_prompt
        }
    
    try:
        response = requests.post(url=url, json=data)
        response.raise_for_status()
        api_result = response.json()
        print("==多模态接口返回结果==:")
        print(api_result)

        has_target = api_result['prediction'].strip().lower() == '是'
        return DetectionResponse(
            has_target=has_target,
            details=[],
            message=message_true if has_target else message_false
        )
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"多模态接口调用失败: {str(e)}")
    except KeyError as e:
        raise HTTPException(status_code=500, detail=f"接口返回格式异常: {str(e)}")

app.include_router(mm_api_router)

# === 启动入口 ===
if __name__ == "__main__":
    import uvicorn
    # 🔍🔍 打印所有已注册的路由路径（调试用）
    print("\n🔍🔍 服务启动后，已注册的路由列表：")
    for route in app.routes:
        print(f"  - {route.path}")
    uvicorn.run(app, host="0.0.0.0", port=7777)