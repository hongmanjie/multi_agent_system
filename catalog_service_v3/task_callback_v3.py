from fastapi import FastAPI, Request
from pydantic import BaseModel
import logging
from datetime import datetime
import json
from typing import Optional, Any
import aiohttp
import asyncio

# Import configuration

# ===================== 配置项 =====================

# 回调服务端口
CALLBACK_SERVER_PORT = 10001
# 可选：是否将回调数据持久化到JSON文件
ENABLE_DATA_PERSISTENCE = False
# 持久化文件路径
PERSISTENCE_FILE = "catalog_callback_records.json"

# 分析接口配置
ANALYSIS_BASE_URL = "http://localhost:8081"
ANALYSIS_CALLBACK_URL = f"{ANALYSIS_BASE_URL}/api/v1/analysis/callback/result"
# 分析接口回调配置
ANALYSIS_CALLBACK_RETRY = 3
ANALYSIS_CALLBACK_TIMEOUT = 10
# ==================================================

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler()  # 控制台输出
        # logging.FileHandler("catalog_callback.log", encoding="utf-8")  # 写入日志文件
    ]
)
logger = logging.getLogger("CatalogCallbackServer")

# 创建FastAPI应用
app = FastAPI(title="视频编目全局回调服务", version="1.0.0")

# 定义回调数据模型
class CatalogCallbackData(BaseModel):
    taskId: int
    resourceId: int
    fileUrl: str
    status: str  # "completed" 或 "failed"
    progress: int
    startTime: Optional[float] = None
    endTime: Optional[float] = None
    costTime: Optional[float] = None
    result: Optional[Any] = None

# 初始化持久化文件
if ENABLE_DATA_PERSISTENCE:
    try:
        with open(PERSISTENCE_FILE, "r", encoding="utf-8") as f:
            json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        with open(PERSISTENCE_FILE, "w", encoding="utf-8") as f:
            json.dump([], f, ensure_ascii=False, indent=2)
    logger.info(f"已启用回调数据持久化，文件路径：{PERSISTENCE_FILE}")

# 异步发送数据到分析接口
async def send_to_analysis_interface(task_id: int, callback_data: CatalogCallbackData):
    """
    将回调数据转换为分析接口要求的格式，异步发送并处理重试
    """
    if not ANALYSIS_CALLBACK_URL:
        logger.warning(f"任务ID: {task_id} 分析接口地址未配置，跳过发送")
        return
    
    # 1. 格式转换：匹配分析接口的要求
    analysis_request_data = {
        "taskId": task_id,
        "success": callback_data.status == "completed",  # completed=True，failed=False
        "msg": "" if callback_data.status == "completed" else (callback_data.result.get("error") if isinstance(callback_data.result, dict) else "任务执行失败"),
        "result": callback_data.result or {}  # 直接复用原回调的result，匹配接口格式
    }
    
    # 2. 带重试的发送逻辑
    for retry in range(1, ANALYSIS_CALLBACK_RETRY + 1):
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    url=ANALYSIS_CALLBACK_URL,
                    json=analysis_request_data,
                    timeout=aiohttp.ClientTimeout(total=ANALYSIS_CALLBACK_TIMEOUT),
                    headers={"Content-Type": "application/json"}
                ) as response:
                    if response.status == 200:
                        logger.info(f"任务ID: {task_id} 发送到分析接口成功（重试 {retry}/{ANALYSIS_CALLBACK_RETRY}）")
                        logger.debug(f"任务ID: {task_id} 分析接口响应：{await response.text()}")
                        return
                    else:
                        logger.warning(
                            f"任务ID: {task_id} 发送到分析接口失败，状态码：{response.status} "
                            f"(重试 {retry}/{ANALYSIS_CALLBACK_RETRY})"
                        )
        except Exception as e:
            logger.warning(
                f"任务ID: {task_id} 发送到分析接口异常：{str(e)} "
                f"(重试 {retry}/{ANALYSIS_CALLBACK_RETRY})"
            )
        
        # 非最后一次重试，等待1秒后重试
        if retry < ANALYSIS_CALLBACK_RETRY:
            await asyncio.sleep(1)
    
    # 所有重试失败
    logger.error(f"任务ID: {task_id} 发送到分析接口失败，已耗尽 {ANALYSIS_CALLBACK_RETRY} 次重试")

# 全局默认回调接口
@app.post("/catalog-callback", summary="接收视频编目任务全局回调")
async def receive_catalog_callback(callback_data: CatalogCallbackData, request: Request):
    """
    接收原视频编目服务推送的全局回调通知
    处理逻辑：记录日志 → 可选持久化 → 非阻塞发送到分析接口 → 返回200成功响应
    """
    task_id = callback_data.taskId
    
    # 1. 记录核心日志
    logger.info(f"收到全局回调通知 | 任务ID: {task_id} | 状态: {callback_data.status} | 耗时: {callback_data.costTime or 0}秒")

    # 2. 可选：仅将result字段持久化到JSON文件
    if ENABLE_DATA_PERSISTENCE:
        try:
            # 提取核心数据
            new_record = {
                "taskId": task_id,
                "result": callback_data.result or None
            }
            
            # 读取原有记录
            with open(PERSISTENCE_FILE, "r", encoding="utf-8") as f:
                records = json.load(f)
            
            # 添加新记录
            records.insert(0, new_record)
            
            # 写入文件
            with open(PERSISTENCE_FILE, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)
            
            logger.debug(f"任务ID: {task_id} 的result已持久化到文件")
        except Exception as e:
            logger.error(f"任务ID: {task_id} 的result持久化失败 | 异常: {str(e)}")

    # 3. 使用create_task创建异步任务，避免阻塞当前请求
    asyncio.create_task(send_to_analysis_interface(task_id, callback_data))

    # 4. 返回200成功响应
    return {
        "code": 200,
        "msg": "回调接收成功",
        "data": {"taskId": task_id, "callbackTime": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    }

@app.get("/health", summary="服务健康检查")
async def health_check():
    return {
        "code": 200,
        "msg": "回调服务运行正常",
        "data": {
            "service_name": app.title,
            "version": app.version,
            "current_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "analysis_callback_url": ANALYSIS_CALLBACK_URL
        }
    }

# 启动回调服务
if __name__ == "__main__":
    import uvicorn
    logger.info(f"启动视频编目全局回调服务 | 端口: {CALLBACK_SERVER_PORT} | 回调接口: http://localhost:{CALLBACK_SERVER_PORT}/catalog-callback")
    logger.info(f"分析接口地址配置为: {ANALYSIS_CALLBACK_URL}")
    uvicorn.run(
        app,
        host="0.0.0.0",  # 允许外部访问
        port=CALLBACK_SERVER_PORT,
        log_level="info"
    )