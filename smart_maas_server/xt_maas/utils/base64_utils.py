import base64
import numpy as np
import cv2
import io
from PIL import Image

def image_to_base64(image_path):
    """
    将图片转为 Base64流
    :param image_path: 图片路径
    :return:
    """
    with open(image_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode('utf-8')  # base64编码
    return base64_data

def base64_to_img(image_base64):
    """将base64转换为PIL和OpenCV图像"""
    try:
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        image_data = base64.b64decode(image_base64)
        pil_image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        return pil_image
        
    except Exception as e:
        raise ValueError(f"Base64图像解码失败: {e}")