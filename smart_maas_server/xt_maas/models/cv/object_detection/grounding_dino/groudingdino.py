import cv2
from PIL import Image
import numpy as np
import torch
from typing import Tuple, Optional, List, Dict, Any

from transformers import (
    AutoProcessor, 
    AutoModelForZeroShotObjectDetection
)

class GroundingDINO:
    def __init__(self, model_path, text_prompt,  # text_prompt现在是List[List[str]]
                box_threshold=0.45, text_threshold=0.25, 
                size=[1200, 800], max_size=None,
                device="cuda:0", enable_cuda=True):
        # 设备设置
        self.device = device if enable_cuda and torch.cuda.is_available() else "cpu"
        
        # 加载HuggingFace模型和处理器
        try:
            self.processor = AutoProcessor.from_pretrained(model_path)
            self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_path).to(self.device)
        except Exception as e:
            raise RuntimeError(f"Failed to initialize HuggingFace model: {str(e)}")
        
        # 配置参数
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

        print("111111111111111111111")
        print("box_threshold:", self.box_threshold)
        print("text_threshold:", self.text_threshold)
        
        # 直接使用传入的text_prompt作为text_labels
        self.text_labels = text_prompt  # 现在直接使用List[List[str]]格式
        
        # 保留原始参数但不使用
        self.size = size
        self.max_size = max_size

    def predict(self, image):
        """
        执行目标检测
        
        参数:
            image: 输入图像 (PIL Image 或 numpy 数组)
            
        返回:
            boxes_filt: 检测框坐标 (绝对坐标)
            pred_phrases: 预测标签
            scores: 置信度分数
        """
        # 处理不同类型的图像输入
        if isinstance(image, Image.Image):
            # 如果已经是 PIL Image，直接使用（已经是 RGB 格式）
            image = image
        elif isinstance(image, np.ndarray):
            # 如果是 numpy 数组，假设是 BGR 格式，转换为 RGB 格式的 PIL Image
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"不支持的图像类型: {type(image)}，期望 PIL.Image 或 numpy.ndarray")
        
        try:
            # 准备输入
            inputs = self.processor(
                images=image, 
                text=self.text_labels, 
                return_tensors="pt"
            ).to(self.device)
            
            # 执行推理
            with torch.no_grad():
                outputs = self.model(**inputs)
            
            # 后处理结果
            post_results = self.processor.post_process_grounded_object_detection(
                outputs,
                inputs.input_ids,
                threshold=self.box_threshold,
                text_threshold=self.text_threshold,
                target_sizes=[image.size[::-1]]  # 保持原始尺寸
            )
            
            # 提取结果
            post_result = post_results[0]
            boxes_filt = []
            pred_phrases = []
            scores = []
            
            # 修改这里：使用 text_labels 而不是 labels
            for box, score, label in zip(post_result["boxes"], post_result["scores"], post_result["text_labels"]):
                if label == "armored vehicle":
                    if score >= 0.8:
                        # 返回绝对坐标
                        boxes_filt.append(box.tolist())
                        pred_phrases.append(label)
                        scores.append(score.item())  # 添加置信度分数

                elif score >= self.box_threshold:
                    # 返回绝对坐标
                    boxes_filt.append(box.tolist())
                    pred_phrases.append(label)
                    scores.append(score.item())  # 添加置信度分数
            
            # return boxes_filt, pred_phrases, scores
            result = {}
            result["boxes"] = boxes_filt
            result["labels"] = pred_phrases
            result["scores"] = scores
            return result

        except Exception as e:
            raise RuntimeError(f"Inference failed: {str(e)}")