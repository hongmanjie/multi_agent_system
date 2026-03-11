# maas_orchestrator.py
import requests
import base64
from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
import logging

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
            TaskType.FALL_DETECTION: TaskConfig(
                task_type=TaskType.FALL_DETECTION,
                min_persons=1,
                vl_question="请仔细观察这张图片，判断画面中是否有人跌倒。如果有人跌倒请回答'是'，如果没有人跌倒请回答'否'。请只回答'是'或'否'，不要提供其他解释。",
                positive_keywords=["是", "yes", "true"],
                negative_keywords=["否", "no", "false"]
            ),
            TaskType.FIGHT_DETECTION: TaskConfig(
                task_type=TaskType.FIGHT_DETECTION,
                min_persons=1,
                vl_question="请仔细观察这张图片，判断画面中是否有打架斗殴行为。如果有打架斗殴行为请回答'是'，如果没有打架斗殴行为请回答'否'。请只回答'是'或'否'，不要提供其他解释。",
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
    
    def yolo_predict(self, image_path: str, config: Dict[str, Any], request_id: str) -> Optional[Dict]:
        """YOLO目标检测"""
        b64 = self.image_to_base64(image_path)
        data = {
            'request_id': request_id,
            'image': b64,
            **config
        }
        
        url = f"{self.base_url}/{ModelType.CV.value}/yolo/predict"
        
        try:
            response = requests.post(url=url, json=data, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            logger.error(f"YOLO请求失败: {e}")
            return None
        except Exception as e:
            logger.error(f"YOLO处理失败: {e}")
            return None
    
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
    
    def count_persons(self, yolo_result: Dict) -> int:
        """统计YOLO检测结果中的人物数量"""
        if not yolo_result or "prediction" not in yolo_result:
            return 0
        
        try:
            boxes = yolo_result["prediction"]["boxes"]
            labels = yolo_result["prediction"]["labels"]
            scores = yolo_result["prediction"]["scores"]
            
            person_count = 0
            for i, label in enumerate(labels):
                if (label == 0 or label == 'person' or label.lower() == 'person') and scores[i] > 0.5:
                    person_count += 1
            
            return person_count
        except Exception as e:
            logger.error(f"统计人物数量失败: {e}")
            return 0
    
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
    
    def execute_task(self, image_path: str, task_type: TaskType, custom_config: Optional[TaskConfig] = None, model_id: str = "qwen2_5_vl") -> Dict[str, Any]:
        """执行指定的AI任务"""
        config = custom_config or self.task_configs.get(task_type)
        if not config:
            raise ValueError(f"未找到任务类型 {task_type} 的配置")
        
        logger.info(f"开始执行 {task_type} 任务，图片: {image_path}")
        
        # 第一步：YOLO目标检测
        logger.info("进行YOLO目标检测...")
        yolo_result = self.yolo_predict(image_path, config.yolo_config, f"{task_type}_yolo_001")
        print(yolo_result)
        print("----------------------")
        
        if yolo_result is None:
            return {
                "success": False,
                "result": False,
                "reason": "目标检测失败",
                "person_count": 0,
                "details": "YOLO检测失败"
            }
        
        # 统计人物数量
        person_count = self.count_persons(yolo_result)
        logger.info(f"检测到人物数量: {person_count}")
        
        # 检查人物数量是否满足要求
        if person_count < config.min_persons:
            reason = f"人物数量不足({person_count}人，需要至少{config.min_persons}人)"
            return {
                "success": True,
                "result": False,
                "reason": reason,
                "person_count": person_count,
                "details": yolo_result
            }
        
        # 第二步：使用VL大模型进行判断
        logger.info("人物数量满足要求，进行VL大模型复核检测...")
        vl_result = self.vl_predict(image_path, config.vl_question, config.vl_config, f"{task_type}_vl_002", model_id)
        
        if vl_result is None:
            return {
                "success": False,
                "result": False,
                "reason": "VL模型检测失败",
                "person_count": person_count,
                "details": yolo_result
            }
        
        # 解析VL模型的回答
        try:
            raw_answer = self.extract_answer_text(vl_result)
            answer = (raw_answer or "").strip().lower()
            logger.info(f"VL模型回答: {answer}")
            
            # 判定结果
            has_positive = any(keyword in answer for keyword in config.positive_keywords)
            has_negative = any(keyword in answer for keyword in config.negative_keywords)
            
            if has_positive and not has_negative:
                result = True
            elif has_negative and not has_positive:
                result = False
            else:
                # 无法明确判断
                result = False
                logger.warning(f"VL模型回答不明确: {answer}")
            
            return {
                "success": True,
                "result": result,
                "reason": "VL模型判断",
                "person_count": person_count,
                "vl_answer": answer,
                "yolo_details": yolo_result,
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
    orchestrator = MaaSOrchestrator("http://127.0.0.1:8010")
    
    # 执行跌倒检测 - 使用qwen2_5_vl模型
    # result = orchestrator.execute_task("140.jpg", TaskType.FALL_DETECTION, model_id="qwen2_5_vl")
    # print(f"跌倒检测结果(qwen2_5_vl): {result}")
    
    # 执行跌倒检测 - 使用minicpmv_4_5模型
    result = orchestrator.execute_task("140.jpg", TaskType.FALL_DETECTION, model_id="minicpmv_4_5")
    print(f"跌倒检测结果(minicpmv_4_5): {result}")
    
    # # 执行打架检测 - 使用qwen2_5_vl模型
    # result = orchestrator.execute_task("140.jpg", TaskType.FIGHT_DETECTION, model_id="qwen2_5_vl")
    # print(f"打架检测结果(qwen2_5_vl): {result}")
    
    # # 执行打架检测 - 使用minicpmv_4_5模型
    # result = orchestrator.execute_task("140.jpg", TaskType.FIGHT_DETECTION, model_id="minicpmv_4_5")
    # print(f"打架检测结果(minicpmv_4_5): {result}")