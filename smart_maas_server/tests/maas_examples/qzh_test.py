# maas_orchestrator.py
import requests
import base64
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
import logging
import os
import cv2

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelType(str, Enum):
    CV = "cv"
    MULTI_MODAL = "multi_modal"
    NLP = "nlp"

class TaskType(str, Enum):
    FALL_DETECTION = "fall_detection"
    FIGHT_DETECTION = "fight_detection"
    PERSON_DETECTION = "person_detection"
    # 可以扩展其他任务类型

class TaskConfig(BaseModel):
    """任务配置"""
    task_type: TaskType
    yolo_config: Dict[str, Any] = Field(default_factory=lambda: {
        "nms_threshold": 0.4,
        "score_threshold": 0.5,
        "roi": []
    })
    vl_config: Dict[str, Any] = Field(default_factory=lambda: {
        "temperature": 0.1
    })
    min_persons: int = 1
    vl_question: str
    positive_keywords: List[str] = Field(default_factory=lambda: ["是", "yes", "true"])
    negative_keywords: List[str] = Field(default_factory=lambda: ["否", "no", "false"])

class MaaSOrchestrator:
    """通用的MaaS服务编排器"""
    
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.task_configs = self._load_default_configs()
    
    def _load_default_configs(self) -> Dict[TaskType, TaskConfig]:
        """加载默认任务配置"""
        return {
            TaskType.PERSON_DETECTION: TaskConfig(
                task_type=TaskType.PERSON_DETECTION,
                min_persons=1,
                vl_question="请仔细观察这张图片，判断画面中的人数。回答时直接输出人数，如，”图像中有三人“，没有检测到人时输出：”图像中没有人“，无需其它赘述。",
                positive_keywords=["是", "yes", "true"],
                negative_keywords=["否", "no", "false"]
            )
        }
    
    def image_to_base64(self, image_path: str) -> str:
        """将图片转为Base64流"""
        try:
            with open(image_path, "rb") as file:
                base64_data = base64.b64encode(file.read()).decode('utf-8')
            return base64_data
        except Exception as e:
            logger.error(f"图片转Base64失败: {e}")
            raise
    
    def vl_predict(self, image_path: str, question: str, config: Dict[str, Any], request_id: str, model_id: str = "qwen2_5_vl") -> Optional[Dict]:
        """视觉语言大模型预测"""
        b64 = self.image_to_base64(image_path)
        data = {
            'request_id': request_id,
            'image': b64,
            'text': question,
            **config
        }
        
        url = f"{self.base_url}/{ModelType.MULTI_MODAL.value}/{model_id}/predict"
        
        try:
            response = requests.post(url=url, json=data, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"VL模型请求失败: {e}")
            return None
        except Exception as e:
            logger.error(f"VL模型处理失败: {e}")
            return None
    
    def extract_answer_text(self, vl_raw: Any) -> str:
        """提取VL模型的回答文本"""
        # 直接是字符串
        if isinstance(vl_raw, str):
            return vl_raw
        # 字典结构
        if isinstance(vl_raw, dict):
            # 常见结构: { prediction: { text: "..." } }
            prediction = vl_raw.get("prediction")
            # 变体: prediction 直接是字符串
            if isinstance(prediction, str):
                return prediction
            if isinstance(prediction, dict):
                text_value = prediction.get("text")
                if isinstance(text_value, str):
                    return text_value
                if isinstance(text_value, list) and text_value and isinstance(text_value[0], str):
                    return text_value[0]
                # 变体: { prediction: { answer: "..." } }
                answer_value = prediction.get("answer")
                if isinstance(answer_value, str):
                    return answer_value
            # 变体: prediction 是列表
            if isinstance(prediction, list) and prediction and isinstance(prediction[0], dict):
                text_value = prediction[0].get("text")
                if isinstance(text_value, str):
                    return text_value
                answer_value = prediction[0].get("answer")
                if isinstance(answer_value, str):
                    return answer_value
            # 顶层直接有 text
            text_value = vl_raw.get("text")
            if isinstance(text_value, str):
                return text_value
        # 其他类型
        return str(vl_raw)
    
    def parse_person_count(self, answer: str) -> Optional[int]:
        """从VL回答中解析人物数量"""
        try:
            # 尝试从回答中提取数字
            import re
            numbers = re.findall(r'\d+', answer)
            if numbers:
                return int(numbers[0])
            
            # 检查常见的中文数字表达
            chinese_numbers = {
                '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, 
                '五': 5, '六': 6, '七': 7, '八': 8, '九': 9, '十': 10
            }
            for char, num in chinese_numbers.items():
                if char in answer:
                    return num
            
            # 检查是否包含"没有人"或类似表达
            if any(keyword in answer for keyword in ["没有人", "没有检测到人", "图像中没有人", "无人"]):
                return 0
                
            return None
        except Exception as e:
            logger.error(f"解析人物数量失败: {e}")
            return None
    
    def execute_task(self, image_path: str, task_type: TaskType, custom_config: Optional[TaskConfig] = None, model_id: str = "qwen2_5_vl") -> Dict[str, Any]:
        """执行指定的AI任务"""
        config = custom_config or self.task_configs.get(task_type)
        if not config:
            raise ValueError(f"未找到任务类型 {task_type} 的配置")
        
        logger.info(f"开始执行 {task_type} 任务，图片: {image_path}")
        
        # 使用VL大模型进行判断
        logger.info("进行VL大模型复核检测...")
        vl_result = self.vl_predict(image_path, config.vl_question, config.vl_config, f"{task_type}_vl_002", model_id)
        
        if vl_result is None:
            return {
                "success": False,
                "result": False,
                "reason": "VL模型检测失败",
            }
        
        # 解析VL模型的回答
        try:
            raw_answer = self.extract_answer_text(vl_result)
            answer = (raw_answer or "").strip().lower()
            logger.info(f"VL模型回答: {answer}")
            
            
            # 特殊处理人物检测任务
            if task_type == TaskType.PERSON_DETECTION:
                person_count = self.parse_person_count(answer)
                return {
                    "success": True,
                    "result": person_count if person_count is not None else 0,
                    "reason": "VL模型人物计数",
                    "person_count": person_count,
                    "vl_answer": answer,
                    "vl_details": vl_result
                }
            
        except Exception as e:
            logger.error(f"解析VL模型回答失败: {e}")
            return {
                "success": False,
                "result": False,
                "reason": "解析VL回答失败",
                "person_count": person_count,
                "details": {"yolo": yolo_result, "vl": vl_result}
            }

# 示例使用
if __name__ == "__main__":
    # 初始化编排器
    orchestrator = MaaSOrchestrator("http://32.74.6.119:8000")
    
    file_path = "/workspace/collect/suicide/2025-09-29"
    
    new_path = "/workspace/newpath/"
    
    images = os.listdir(file_path)
    for image in images:
      image_path = os.path.join(file_path, image)
      
      cv_image = cv2.imread(image_path)
      
      person_max = 0
      
      # 执行人数检测 - 使用qwen2_5_vl模型
      result_qw = orchestrator.execute_task(image_path, TaskType.PERSON_DETECTION, model_id="qwen2_5_vl")
      print(f"人数检测结果(qwen2_5_vl): {result_qw}")
      
      # 执行人数检测 - 使用minicpmv_4_5模型
      result_mini = orchestrator.execute_task(image_path, TaskType.PERSON_DETECTION, model_id="minicpmv_4_5")
      print(f"人数检测结果(minicpmv_4_5): {result_mini}")
      
      person_num_qw = result_qw.get("result")
      person_num_mini = result_mini.get("result")
      
      print(person_num_qw)
      print(person_num_mini)
      
      if person_num_qw >= person_num_mini:
        person_max = person_num_qw
      else:
        person_max = person_num_mini
      
      
      save_path = os.path.join(new_path, str(person_max), image)
      
      cv2.imwrite(save_path, cv_image)
      
      print("图像保存成功!")
      
      
      
      
      