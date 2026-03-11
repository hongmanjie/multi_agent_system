import os
import cv2
import numpy as np
import onnxruntime as ort
from typing import List, Dict, Any, Tuple, Optional
import logging
import time
from PIL import Image
import torchvision.transforms as T
from datetime import datetime

logger = logging.getLogger("YOLOv12_ONNX_PERSON")

class YOLOv12_ONNX_PERSON:
    """YOLOv12 ONNX推理类 - 适配您的模型"""
    
    def __init__(self, model_path: str, device: str = "cpu", **kwargs):
        self.model_path = model_path
        self.device = device
        sess_options = ort.SessionOptions()
        sess_options.enable_mem_pattern = False
        sess_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.enable_cpu_mem_arena = False
        sess_options.enable_mem_reuse = True

        # 检查模型文件
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"模型文件不存在: {model_path}")
        
        # 设置执行提供者
        providers = ["CPUExecutionProvider"]
        if device.lower() == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        
        # 创建推理会话
        self.session = ort.InferenceSession(model_path, providers=providers, sess_options=sess_options)
        
        # 获取输入输出信息
        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()
        
        # 记录模型信息
        logger.info("YOLOv12模型信息:")
        for i, inp in enumerate(self.inputs):
            logger.info(f"  输入 {i}: 名称={inp.name}, 形状={inp.shape}, 类型={inp.type}")
        for i, out in enumerate(self.outputs):
            logger.info(f"  输出 {i}: 名称={out.name}, 形状={out.shape}, 类型={out.type}")
        
        # 从输入形状获取模型尺寸
        self.size = 640  # 根据您的模型信息
        
        # COCO数据集类别名称
        self.class_names = [
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
            "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
            "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
            "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
            "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
            "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"
        ]
        self.class_map = {i: name for i, name in enumerate(self.class_names)}
        
        # 配置参数
        self.conf_threshold = kwargs.get("obj_thresh", 0.5)
        self.iou_threshold = kwargs.get("nms_thresh", 0.7)


    def get_class_name(self, class_id: int) -> str:
        """根据类别ID获取类别名称"""
        return self.class_map.get(class_id, f"class_{class_id}")
    
        
    def resize_with_aspect_ratio(self, image: Image.Image, size: int):
        """保持宽高比调整图像大小"""
        original_width, original_height = image.size
        ratio = min(size / original_width, size / original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        image = image.resize((new_width, new_height), Image.BILINEAR)

        new_image = Image.new("RGB", (size, size))
        new_image.paste(image, ((size - new_width) // 2, (size - new_height) // 2))
        return new_image, ratio, (size - new_width) // 2, (size - new_height) // 2
        
        
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


    def predict(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        """执行推理预测"""
        # 读取图像
        start_time = time.time()
        try:
            # 获取参数
            roi =  kwargs.get("roi")
            conf_threshold = kwargs.get("score_threshold")
            iou_threshold = kwargs.get("nms_threshold")
            original_width, original_height = image.size
            logger.debug(f"📐 原始图像尺寸: {original_width}x{original_height}")

            # 预处理
            resized_image, ratio, pad_w, pad_h = self.resize_with_aspect_ratio(image, self.size)
            logger.info(f"🔄 调整后尺寸: {resized_image.size}x{resized_image.size}, 比例: {ratio:.4f}, 填充: ({pad_w}, {pad_h})")
            
            # 转换为模型输入格式
            transforms = T.Compose([T.ToTensor()])
            input_tensor = transforms(resized_image).unsqueeze(0)

            # 执行推理 - 使用正确的输入名称 'images'
            outputs = self.session.run(['output0'], {'images': input_tensor.numpy()})
            
            # 处理输出
            output_tensor = outputs[0] if isinstance(outputs, list) else outputs  # 形状: (1, 84, 8400)
            predictions_tensor = output_tensor[0]
            detections = []
            valid_count = 0
            #[8400, 84]
            predictions_tensor = predictions_tensor.transpose(1, 0)
            for i in range(predictions_tensor.shape[0]):
                detection = predictions_tensor[i]
                x_center, y_center, w, h, confidence = detection
                if confidence < conf_threshold:
                    continue
                # 将中心坐标转换为边界框坐标
                x1 = float((x_center - w/2 - pad_w) / ratio)
                y1 = float((y_center - h/2 - pad_h) / ratio)
                x2 = float((x_center + w/2 - pad_w) / ratio)
                y2 = float((y_center + h/2 - pad_h) / ratio)
                
                # 确保边界框在图像范围内
                x1 = max(0, min(x1, original_width))
                y1 = max(0, min(y1, original_height))
                x2 = max(0, min(x2, original_width))
                y2 = max(0, min(y2, original_height))

                adjusted_bb = [x1, y1, x2, y2]

                in_roi = False
                
                if roi:
                    for roi_area in roi:
                        if self.is_box_in_roi(adjusted_bb, roi_area):
                            in_roi = True
                            break
                else:
                    in_roi = True
                
                if in_roi:
                    detections.append({
                        "bbox": adjusted_bb,
                        "class": "person",
                        "conf": float(confidence)
                    })
                    
            logger.info(f"🎯 任务'person'相关检测: {len(detections)}个")

            # 应用NMS过滤重叠框
            if detections:
                boxes = [det['bbox'] for det in detections]
                scores = [det['conf'] for det in detections]

                # 使用OpenCV NMS
                indices = cv2.dnn.NMSBoxes(
                    boxes, scores, conf_threshold, iou_threshold
                )

                if len(indices) > 0:
                    indices = indices.flatten()
                    filtered_detections = [detections[i] for i in indices]
                    detections = filtered_detections
                    valid_count = len(detections)
                    logger.info(f"✅ NMS后保留: {valid_count}个检测")

            result = {
                "count": valid_count,
                "detections": detections,
                "success": True,
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            result = {
                "detections": [],
                "success": False,
                "error": str(e),
                "processing_time": time.time() - start_time,
                "timestamp": datetime.now()
            }
        
        return result

    def __call__(self, image: np.ndarray, **kwargs) -> Dict[str, Any]:
        return self.predict(image, **kwargs)

# 在您的MaaS编排器中使用
def yolo_predict(self, image_path: str, config: Dict[str, Any], request_id: str) -> Optional[Dict]:
    """YOLO目标检测 - 修正版本"""
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"无法读取图像: {image_path}")
            return None
        
        # 使用修正后的YOLOv12类进行推理
        yolo_model = YOLOv12_ONNX(self.yolo_model_path)  # 需要设置模型路径
        result = yolo_model.predict(image, **config)
        
        return result
        
    except Exception as e:
        logger.error(f"YOLO预测失败: {e}")
        return None