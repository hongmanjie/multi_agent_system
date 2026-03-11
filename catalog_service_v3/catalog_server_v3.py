import asyncio
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from enum import Enum
import logging
import time
from datetime import datetime
from fastapi import FastAPI
from pydantic import BaseModel
import aiohttp
from concurrent.futures import ProcessPoolExecutor
from auto_catalog_v3 import generate_analysis_file
from contextlib import asynccontextmanager
import tempfile
import os
import aiofiles


# 全局默认回调地址
GLOBAL_CALLBACK_URL = "http://localhost:10001/catalog-callback"
# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 临时文件配置（可根据实际情况调整）
TEMP_DIR = "./temp_videos"
# 确保临时目录存在
os.makedirs(TEMP_DIR, exist_ok=True)


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"          # 等待队列中，待执行
    RUNNING = "running"          # 运行队列中，正在执行
    COMPLETED = "completed"      # 已完成
    FAILED = "failed"            # 已失败


@dataclass
class CatalogTask:
    """编目任务数据类"""
    task_id: int
    resource_id: int
    file_url: str
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    result: Optional[Any] = None
    
    @property
    def callback_url(self) -> Optional[str]:
        """获取全局默认回调地址"""
        return GLOBAL_CALLBACK_URL if GLOBAL_CALLBACK_URL else None


async def download_video_to_local(file_url: str, task_id: int) -> Optional[str]:
    """
    从HTTP URL下载视频到本地临时文件
    :param file_url: 视频HTTP地址
    :param task_id: 任务ID（用于命名临时文件）
    :return: 本地临时文件路径，下载失败返回None
    """
    try:
        # 构造临时文件名（保留原文件后缀，避免编目函数不识别）
        file_suffix = os.path.splitext(file_url)[-1]
        temp_file_path = os.path.join(TEMP_DIR, f"task_{task_id}_{int(time.time())}{file_suffix}")
        
        logger.info(f"Task {task_id} starting to download video: {file_url} -> {temp_file_path}")
        
        # 异步下载文件（分块写入，避免大文件占用过多内存）
        async with aiohttp.ClientSession() as session:
            async with session.get(file_url, timeout=aiohttp.ClientTimeout(total=300)) as response:
                if response.status != 200:
                    logger.error(f"Task {task_id} download failed, HTTP status: {response.status}")
                    return None
                
                async with aiofiles.open(temp_file_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(1024 * 1024):  # 1MB分块
                        await f.write(chunk)
        
        # 验证文件是否下载成功（是否存在且大小大于0）
        if os.path.exists(temp_file_path) and os.path.getsize(temp_file_path) > 0:
            logger.info(f"Task {task_id} video downloaded successfully, file size: {os.path.getsize(temp_file_path)} bytes")
            return temp_file_path
        else:
            logger.error(f"Task {task_id} download failed, temp file is empty or not exists")
            # 清理空文件
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            return None
    
    except Exception as e:
        logger.error(f"Task {task_id} download exception: {str(e)}")
        return None


def clean_temp_file(temp_file_path: str, task_id: int):
    """
    清理本地临时文件
    :param temp_file_path: 临时文件路径
    :param task_id: 任务ID
    """
    try:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info(f"Task {task_id} temp file cleaned: {temp_file_path}")
    except Exception as e:
        logger.warning(f"Task {task_id} temp file clean failed: {str(e)}")


class VideoCatalogQueue:
    """视频编目队列管理类"""
    
    def __init__(self, max_concurrent_tasks: int = 3):
        self.max_concurrent_tasks = max_concurrent_tasks
        self.pending_queue: List[CatalogTask] = []
        self.running_queue: List[CatalogTask] = []
        self.lock = threading.RLock()
        self.task_map: Dict[int, CatalogTask] = {}
        self.callback_retry_times = 3
        self.callback_timeout = 10
        self._print_queue_running = False
        self.process_pool = ProcessPoolExecutor(max_workers=3)
    
    def add_task(self, task: CatalogTask) -> bool:
        """添加任务到等待队列"""
        with self.lock:
            if task.task_id in self.task_map:
                existing_task = self.task_map[task.task_id]
                logger.warning(
                    f"Task ID {task.task_id} already exists, current status: {existing_task.status.value}"
                )
                return False
            
            task.status = TaskStatus.PENDING
            task.progress = 0
            self.pending_queue.append(task)
            self.task_map[task.task_id] = task
            
            logger.info(
                f"Added task {task.task_id} to pending queue. "
                f"Pending queue size: {len(self.pending_queue)}, Running queue size: {len(self.running_queue)}"
            )
            
            self._try_start_next_task()
            return True
    
    def _try_start_next_task(self):
        """填充运行队列空闲槽位"""
        with self.lock:
            while len(self.running_queue) < self.max_concurrent_tasks and len(self.pending_queue) > 0:
                next_task = self.pending_queue.pop(0)
                next_task.status = TaskStatus.RUNNING
                next_task.start_time = time.time()
                next_task.progress = 0
                self.running_queue.append(next_task)
                self.task_map[next_task.task_id] = next_task
                
                logger.info(
                    f"Task {next_task.task_id} moved from pending to running. "
                    f"Pending queue: {len(self.pending_queue)}, Running queue: {len(self.running_queue)}/{self.max_concurrent_tasks}"
                )
                
                asyncio.create_task(self._execute_task(next_task))
    
    async def _send_callback(self, task: CatalogTask):
        """异步发送回调请求"""
        callback_url = task.callback_url
        if not callback_url:
            logger.info(f"Task {task.task_id} has no callback url configured, skip callback")
            return
        
        callback_data = {
            "taskId": task.task_id,
            "resourceId": task.resource_id,
            "fileUrl": task.file_url,
            "status": task.status.value,
            "progress": task.progress,
            "startTime": task.start_time,
            "endTime": task.end_time,
            "costTime": round(task.end_time - task.start_time, 2) if task.start_time and task.end_time else 0,
            "result": task.result
        }
        
        for retry in range(1, self.callback_retry_times + 1):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url=callback_url,
                        json=callback_data,
                        timeout=aiohttp.ClientTimeout(total=self.callback_timeout),
                        headers={"Content-Type": "application/json"}
                    ) as response:
                        if response.status == 200:
                            logger.info(f"Task {task.task_id} callback success (retry {retry}/{self.callback_retry_times})")
                            logger.debug(f"Task {task.task_id} callback response: {await response.text()}")
                            return
                        else:
                            logger.warning(
                                f"Task {task.task_id} callback failed, status code: {response.status} "
                                f"(retry {retry}/{self.callback_retry_times})"
                            )
            except Exception as e:
                logger.warning(
                    f"Task {task.task_id} callback exception: {str(e)} "
                    f"(retry {retry}/{self.callback_retry_times})"
                )
            
            if retry < self.callback_retry_times:
                await asyncio.sleep(1)
        
        logger.error(f"Task {task.task_id} callback failed after {self.callback_retry_times} retries")
    
    async def _execute_task(self, task: CatalogTask):
        """执行单个编目任务"""
        temp_file_path = None  # 初始化临时文件路径
        try:
            logger.info(f"Starting to execute task {task.task_id}, file: {task.file_url}")
            
            # 步骤1：下载视频到本地临时文件
            temp_file_path = await download_video_to_local(task.file_url, task.task_id)
            if not temp_file_path:
                raise Exception(f"Failed to download video from {task.file_url}")
            
            # 步骤2：调用编目函数处理本地临时文件
            try:
                logger.info(f"Submitting task {task.task_id} to process pool for catalog analysis (local file: {temp_file_path})")
                loop = asyncio.get_running_loop()
                results = await loop.run_in_executor(self.process_pool, generate_analysis_file, temp_file_path, 25, None, task.task_id, task.resource_id)
                
                with self.lock:
                    task.status = TaskStatus.COMPLETED
                    task.progress = 100
                    task.result = results
                    task.end_time = time.time()
                
                logger.info(f"Task {task.task_id} executed successfully, result saved")
                
            except Exception as e:
                error_msg = f"Task execution failed: {str(e)}"
                logger.error(error_msg)
                
                with self.lock:
                    task.status = TaskStatus.FAILED
                    task.progress = 0
                    task.result = {"error": error_msg}
                    task.end_time = time.time()
            
        except Exception as e:
            error_msg = f"Unexpected error in task execution: {str(e)}"
            logger.error(error_msg)
            
            with self.lock:
                task.status = TaskStatus.FAILED
                task.progress = 0
                task.result = {"error": error_msg}
                task.end_time = time.time()
        finally:
            # 步骤3：无论执行成功/失败，都清理本地临时文件
            if temp_file_path:
                clean_temp_file(temp_file_path, task.task_id)
            
            with self.lock:
                if task in self.running_queue:
                    self.running_queue.remove(task)
                    logger.info(
                        f"Task {task.task_id} removed from running queue. "
                        f"Running queue size: {len(self.running_queue)}/{self.max_concurrent_tasks}"
                    )
                
                self.task_map[task.task_id] = task
                self._try_start_next_task()
            
            asyncio.create_task(self._send_callback(task))
    
    def get_task(self, task_id: int) -> Optional[CatalogTask]:
        """查询任务状态"""
        with self.lock:
            for task in self.pending_queue:
                if task.task_id == task_id:
                    logger.debug(f"Task {task_id} found in pending queue, status: {task.status.value}")
                    return task
            
            for task in self.running_queue:
                if task.task_id == task_id:
                    logger.debug(f"Task {task_id} found in running queue, status: {task.status.value}")
                    return task
            
            if task_id in self.task_map:
                logger.debug(f"Task {task_id} found in task map, status: {self.task_map[task_id].status.value}")
                return self.task_map.get(task_id)
            
            logger.warning(f"Task {task_id} not found in any queue or task map")
            return None
    
    async def print_queue_status_periodically(self):
        """定时打印队列状态"""
        self._print_queue_running = True
        logger.info("Started periodic queue status printing (interval: 10 seconds)")
        
        while self._print_queue_running:
            try:
                with self.lock:
                    pending_task_ids = [task.task_id for task in self.pending_queue]
                    pending_size = len(pending_task_ids)
                    running_task_ids = [task.task_id for task in self.running_queue]
                    running_size = len(running_task_ids)
                    max_running = self.max_concurrent_tasks
                
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.info(
                    f"===== Queue Status [{current_time}] ====="
                    f"\nPending Queue: size={pending_size}, task_ids={pending_task_ids}"
                    f"\nRunning Queue: size={running_size}/{max_running}, task_ids={running_task_ids}"
                    f"\n=========================================="
                )
                
                await asyncio.sleep(10)
            
            except Exception as e:
                logger.error(f"Error in periodic queue printing: {str(e)}")
                await asyncio.sleep(10)
    
    def stop_periodic_print(self):
        """停止定时打印队列状态"""
        self._print_queue_running = False
        logger.info("Stopped periodic queue status printing")
    
    def shutdown(self):
        """关闭资源，包括进程池"""
        logger.info("Shutting down VideoCatalogQueue resources...")
        self.stop_periodic_print()
        
        # 清空待处理队列
        with self.lock:
            if self.pending_queue:
                logger.info(f"Clearing pending queue: {len(self.pending_queue)} tasks")
                self.pending_queue.clear()
        
        # 关闭进程池
        if hasattr(self, 'process_pool'):
            logger.info("Shutting down process pool...")
            self.process_pool.shutdown(wait=True, cancel_futures=True)  # 等待现有任务完成，取消未完成的任务
            logger.info("Process pool shutdown completed")
        
        logger.info("VideoCatalogQueue shutdown completed")


# 存储所有创建的异步任务
running_tasks = []

# FastAPI 应用生命周期管理
@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI新版本生命周期管理"""
    # 创建并存储定时打印队列状态的任务
    print_task = asyncio.create_task(catalog_queue.print_queue_status_periodically())
    running_tasks.append(print_task)
    yield
    
    # 取消所有正在运行的异步任务
    logger.info("Cancelling all running tasks...")
    for task in running_tasks:
        if not task.done():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                logger.info(f"Task cancelled: {task.get_name()}")
    
    # 关闭资源
    catalog_queue.shutdown()
    logger.info("Application shutdown completed")

# 初始化FastAPI应用
app = FastAPI(title="视频编目队列管理服务", version="1.0.0", lifespan=lifespan)
catalog_queue = VideoCatalogQueue(max_concurrent_tasks=3)


# 通用响应模型
class ApiResponse(BaseModel):
    code: int
    msg: str
    data: dict = {}


# 请求模型
class CreateTaskRequest(BaseModel):
    taskId: int
    resourceId: int
    fileUrl: str


@app.post("/create-task")
async def create_task(request: CreateTaskRequest):
    """
    创建任务接口（严格对齐API文档，返回空data）
    """
    try:
        task = CatalogTask(
            task_id=request.taskId,
            resource_id=request.resourceId,
            file_url=request.fileUrl
        )
        print(f"Received task: {task.task_id}, resource_id: {task.resource_id}, file_url: {task.file_url}")
        
        success = catalog_queue.add_task(task)
        
        if success:
            return ApiResponse(code=200, msg="success", data={})
        else:
            return ApiResponse(code=400, msg=f"Task ID {request.taskId} already exists", data={})
    
    except Exception as e:
        error_msg = f"Failed to create task: {str(e)}"
        logger.error(error_msg)
        return ApiResponse(code=500, msg=error_msg, data={})


@app.get("/task/{task_id}")
async def get_task_status(task_id: int):
    """
    查询任务状态接口（严格对齐API文档，data仅返回大写status）
    """
    try:
        task = catalog_queue.get_task(task_id)
        if not task:
            return ApiResponse(code=404, msg="Task not found", data={})
        
        task_data = {
            "status": task.status.value.upper()
        }
        
        return ApiResponse(
            code=200,
            msg="success",
            data=task_data
        )

    except Exception as e:
        error_msg = f"Failed to get task status: {str(e)}"
        logger.error(error_msg)
        return ApiResponse(
            code=500,
            msg=error_msg,
            data={}
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)