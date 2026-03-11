from typing import List, Dict, Any, Optional
import os
import sys
import time
import logging
from datetime import datetime
from PIL import Image
import numpy as np
import torch
import cv2

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

logger = logging.getLogger("SAM3Model")


class SAM3Model:
    """SAM3 模型的包装类，适配 MaaS 的 CV 模型接口"""
    
    def __init__(self, model_path: str, device: str = "cuda", enable_cuda: bool = True, **kwargs):
        """
        初始化 SAM3 模型
        
        Args:
            model_path: SAM3 模型文件路径
            device: 运行设备，默认为"cuda"
            enable_cuda: 是否启用 CUDA，默认为 True
            **kwargs: 其他配置参数
        """
        self.model_path = model_path
        self.device = device if enable_cuda and torch.cuda.is_available() else "cpu"
        
        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在：{model_path}")
        
        # 导入 SAM3 模型 (sam3 已通过 pip install 安装到 Python 环境)
        try:
            # 加载模型
            logger.info(f"正在加载 SAM3 模型：{model_path}, 设备：{self.device}")
            print(f"正在加载 SAM3 模型：{model_path}, 设备：{self.device}")
            # 如果 device 是具体的 cuda 设备 (如 cuda:2)，需要特殊处理
            # build_sam3_image_model 的_setup_device_and_mode 只处理 "cuda"，不处理 "cuda:X"
            self.model = build_sam3_image_model(checkpoint_path=model_path, device=self.device, load_from_HF=False)
            
            # 确保模型在正确的具体设备上 (如 cuda:2)
            self.model.to(self.device)
            self.model.eval()
            
            self.processor = Sam3Processor(model=self.model, device=self.device)
            logger.info("SAM3 模型加载成功")
            
        except Exception as e:
            logger.error(f"SAM3 模型加载失败：{e}")
            raise RuntimeError(f"Failed to initialize SAM3 model: {str(e)}")
        
        # 配置参数
        self.box_threshold = kwargs.get("box_threshold", 0.5)
        self.text_threshold = kwargs.get("text_threshold", 0.25)
    
    def predict(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """
        执行推理预测
        
        Args:
            image: PIL 图像对象
            categories: 类别列表 (文本提示)
            **kwargs: 推理参数
            
        Returns:
            推理结果，包含检测框、类别、置信度等信息
        """
        start_time = time.time()
        try:
            # 获取检测类别列表
            categories_list = kwargs.get("categories")
            if not categories_list or len(categories_list) == 0:
                raise ValueError("必须提供 categories 参数 (检测类别列表)")
            
            roi = kwargs.get("roi")
            
            # 获取原始图像尺寸
            original_width, original_height = image.size
            logger.debug(f"原始图像尺寸：{original_width}x{original_height}")
            logger.debug(f"当前设备：{self.device}")
            
            # 设置图像 (Sam3Processor 会自动处理设备)
            logger.info(f"开始设置图像...")
            inference_state = self.processor.set_image(image)
            
            # 存储所有检测结果
            all_detections = []
            
            # 对每个类别进行文本提示推理
            for category in categories_list:
                logger.info(f"开始推理类别：{category}")
                
                # 执行文本提示推理
                output = self.processor.set_text_prompt(
                    state=inference_state,
                    prompt=category
                )
                
                # 获取结果
                masks, boxes, scores = output["masks"], output["boxes"], output["scores"]
                
                logger.debug(f"类别 {category} 检测到 {len(masks)} 个目标")
                
                # 处理每个检测结果
                for i, (box, score) in enumerate(zip(boxes, scores)):
                    # 应用置信度阈值
                    if score < self.box_threshold:
                        continue
                    
                    # 坐标格式转换 (如果需要)
                    bbox = box.tolist() if hasattr(box, 'tolist') else list(box)
                    
                    # ROI 检查
                    in_roi = True
                    if roi:
                        in_roi = False
                        for roi_area in roi:
                            if self._is_box_in_roi(bbox, roi_area):
                                in_roi = True
                                break
                    
                    if in_roi:
                        all_detections.append({
                            "bbox": bbox,
                            "class": category,
                            "conf": float(score),
                        })
            
            # 构建返回结果
            result_dict = {
                "count": len(all_detections),
                "detections": all_detections,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now()
            }
            
            logger.info(f"SAM3 推理完成，检测到 {len(all_detections)} 个目标")
            return result_dict
            
        except Exception as e:
            logger.error(f"SAM3 推理失败：{e}", exc_info=True)
            return {
                "detections": [],
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now()
            }
    
    def _is_box_in_roi(self, box: List[float], roi: List[float]) -> bool:
        """
        检查检测框是否在 ROI 区域内
        
        Args:
            box: 检测框 [x1, y1, x2, y2]
            roi: ROI 区域坐标点列表 [x1, y1, x2, y2, ...]
            
        Returns:
            是否在 ROI 内
        """
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        
        # 计算框的中心点
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        # 构建多边形点
        polygon_points = []
        for i in range(0, len(roi), 2):
            if i + 1 < len(roi):
                polygon_points.append([roi[i], roi[i + 1]])
        
        if len(polygon_points) < 3:
            return False
        
        polygon_points = np.array(polygon_points, dtype=np.float32)
        roi_condition = cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0
        
        return roi_condition
    
    def __call__(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """支持直接调用模型实例进行推理"""
        return self.predict(image, **kwargs)
