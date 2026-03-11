from typing import List, Dict, Any, Optional
import os
import sys
import logging
import time
import cv2
from datetime import datetime
from PIL import Image
import numpy as np

# 导入内部封装的RexOmni模块
from .wrapper import RexOmniWrapper
from .tasks import TaskType

logger = logging.getLogger("RexOmniModel")

class RexOmniModel:
    """RexOmni模型的包装类，适配maas_code的CV模型接口"""
    
    def __init__(self, model_path: str, device_map: str = "cuda", **kwargs):
        """
        初始化RexOmni模型
        
        Args:
            model_path: RexOmni模型文件路径
            device: 运行设备，默认为"cuda"
            **kwargs: 其他配置参数
        """
        self.model_path = model_path
        self.device_map = device_map
        
        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 初始化RexOmni模型
        logger.info(f"正在加载RexOmni模型: {model_path}")
        self.model = RexOmniWrapper(model_path=model_path, backend="vllm", device_map=device_map)
        logger.info("RexOmni模型加载成功")
        
        # 配置参数
        self.conf_threshold = kwargs.get("conf_threshold", 0.5)
    
        
    def is_box_in_roi(self, box: List[float], roi: List[float]) -> bool:
        x1, y1, x2, y2 = box
        box_area = (x2 - x1) * (y2 - y1)
        
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        polygon_points = []
        for i in range(0, len(roi), 2):
            if i + 1 < len(roi):
                polygon_points.append([roi[i], roi[i + 1]])
        
        if len(polygon_points) < 3:
            return False
        
        polygon_points = np.array(polygon_points, dtype=np.float32)
        roi_condition = cv2.pointPolygonTest(polygon_points, (center_x, center_y), False) >= 0
        
        return roi_condition
        
        
    def convert_bbox_format(self, bbox: List[float]) -> List[float]:
        """
        将边界框从 [x0, y0, x1, y1] 转换为 [x, y, w, h] 格式
        
        Args:
            bbox: 输入的边界框坐标 [x0, y0, x1, y1]
            
        Returns:
            输出的边界框坐标 [x, y, w, h]
        """
        x0, y0, x1, y1 = bbox
        x = min(x0, x1)
        y = min(y0, y1)
        w = max(x0, x1) - x
        h = max(y0, y1) - y
        return [x, y, w, h]
    
        
    def predict(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """
        执行推理预测
        
        Args:
            image: PIL图像对象
            categories: 类别列表
            **kwargs: 推理参数。
            
        Returns:
            推理结果，包含检测框、类别、置信度等信息
        """
        start_time = time.time()
        try:
            categories_list = kwargs.get("categories")
            roi = kwargs.get("roi")
            # 获取原始图像尺寸
            original_width, original_height = image.size
            logger.debug(f"原始图像尺寸: {original_width}x{original_height}")
            
            # 调用模型推理
            logger.info(f"开始推理，检测类别: {categories_list}")
            results = self.model.inference(
                images=image,
                task=TaskType.DETECTION.value,
                categories=categories_list
            )
            
            # 处理推理结果
            result = results[0] if isinstance(results, list) else results
            if not result.get("success", False):
                error_msg = result.get("error", "Unknown error")
                raise Exception(error_msg)
            
            # 提取检测结果
            predictions = result.get("extracted_predictions", {})
            detections = []
            
            for cls_name in categories_list:
                category = predictions.get(cls_name, [])
                logger.debug(f"类别 {cls_name} 检测到 {len(category)} 个目标")
                
                for c in category:
                    bbox = c.get("coords", [])
                    if len(bbox) != 4:
                        continue
                    
                    in_roi = False
                    if roi:
                      for roi_area in roi:
                        if self.is_box_in_roi(bbox, roi_area):
                          in_roi = True
                          break
                    else:
                      in_roi = True
                    
                    if in_roi:
                      detections.append({
                          "bbox": bbox,
                          "class": cls_name,
                          "conf": 1.0  # RexOmni目前不返回置信度
                      })
            
            # 构建返回结果
            result_dict = {
                "count": len(detections),
                "detections": detections,
                "success": True,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now()
            }
            
            logger.info(f"推理完成，检测到 {len(detections)} 个目标")
            return result_dict
            
        except Exception as e:
            logger.error(f"推理失败: {e}", exc_info=True)
            return {
                "detections": [],
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now()
            }
    
    def __call__(self, image: Image.Image, **kwargs) -> Dict[str, Any]:
        """支持直接调用模型实例进行推理"""
        return self.predict(image, **kwargs)

