# 文件路径应该是: qwen2_5_vl.py (与当前模块同目录下的 qwen2_5_vl.py 文件)
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import torch
# from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import logging
from transformers import (
    AutoProcessor, 
    Qwen2_5_VLForConditionalGeneration)

# from accelerate import infer_auto_device_map, dispatch_model

from xt_maas.utils.device_utils import get_gpu_memory_usage_nvidia_smi

from xt_maas.utils.logger import get_logger

logger = get_logger()

class Qwen2_5VL_Model:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen2.5-VL-2B-Instruct",
        torch_dtype: torch.dtype = torch.float16,
        device_map: str = "auto",
        max_new_tokens: int = 128, 
        min_image_size: int = 256,   # 最小图像尺寸，单位为28x28块
        max_image_size: int = 512,  # 最大图像尺寸，单位为28x28块
    ):
        """
        Qwen2.5-VL 模型初始化
        Args:
            model_path: 模型路径或名称
            torch_dtype: torch 数据类型
            device_map: 指定设备
            max_new_tokens: 最大生成长度
        """
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens

        self.min_pixels = min_image_size * 28 * 28
        self.max_pixels = max_image_size * 28 * 28
        self.model = None
        self.processor = None

        try:
            self._load_model()
            logger.info(f"Qwen2.5-VL 模型成功加载: {self.device}")
        except Exception as e:
            logger.error(f"加载 Qwen2.5-VL 模型失败: {str(e)}")
            raise

    def _load_model(self):
        """加载模型和处理器"""        
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_path,
            torch_dtype=self.torch_dtype,
            device_map=None
        )

        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
        )

        device_map = self.device_map.lower()

        # 显卡配置 
        # if "auto" 自动分配显存
        # if 指定多卡"cuda:0,cuda:1"，则自动计算设备映射
        # else 单卡
        if device_map == "auto":
            if torch.cuda.is_available():
                # print(f"当前可见的 GPU 数量: {torch.cuda.device_count()}")
                # for i in range(torch.cuda.device_count()):
                #     print(f"  CUDA:{i} -> 物理GPU: {torch.cuda.get_device_name(i)}")
                gpu_info = get_gpu_memory_usage_nvidia_smi()
                logger.info(f"当前设备的 GPU 数量: {len(gpu_info)}")
                if gpu_info and len(gpu_info) > 0:
                    best_gpu = max(gpu_info, key=lambda x: x['free_memory_MB'])
                    self.device = f"cuda:{best_gpu['device_id']}"
                    logger.info(f"自动选择剩余显存最大的 GPU: {self.device}，剩余显存: {best_gpu['free_memory_MB']:.2f} MB")
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=self.torch_dtype,
                        device_map=self.device  # 单卡模式
                    )
                else:
                    logger.warning("未能获取 GPU 信息，将使用 CPU 运行")
                    self.device = "cpu"
                    self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                        self.model_path,
                        torch_dtype=self.torch_dtype,
                        device_map=self.device  # 单卡模式
                    )
            else:
                logger.warning("CUDA不可用，将使用CPU运行")
                self.device = "cpu"
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=self.torch_dtype,
                    device_map=self.device  # 单卡模式
                )
        elif "," in device_map:
            # 多卡情况,如 "cuda:0,cuda:1"
            # 配合CUDA_VISIBLE_DEVICES使用
            devices = [d.strip() for d in device_map.split(",")]
            logger.info(f"使用多卡模式，指定设备列表: {devices}")

            # 验证是否是 cuda:X 格式
            if not all(d.startswith("cuda:") for d in devices):
                raise ValueError(f"无效的设备列表: {device_map}。请使用 'cuda:X,cuda:Y' 格式")

            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map="auto"  # HF自动分配到当前可见的 GPU
            )
            self.device = "cuda"
        else: # 单卡情况
            self.device = device_map
            logger.info(f"使用单卡模式，指定设备: {device_map}")
            self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                self.model_path,
                torch_dtype=self.torch_dtype,
                device_map=self.device
            )


    def generate(
        self,
        image: np.ndarray,
        text: str,
        temperature: float = 0.7,
        top_p: float = 0.9
        ) -> str:
        """
        生成文本响应
        
        Args:
            image: ndarray 对象
            text: 文本提示
            temperature: 温度参数
            top_p: top-p采样参数            
        Returns:
            生成的文本
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("模型未加载，请先初始化模型")

        try:
            # 检查输入的image格式
            if image is None:
                raise ValueError("图像输入为空")
            # to Image.Image
            image = Image.fromarray(image)
            messages = [{
                "role": "user",
                "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": text if text else "请详细描述这张图片的内容"}
                ]
                        }]
            # 准备输入
            chat_text = self.processor.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            mm_inputs = self.processor(text=[chat_text], 
                                    images=[image],
                                    padding=True,
                                    return_tensors="pt").to(self.device)

            generated_ids = self.model.generate(
                **mm_inputs, 
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p)

            trimmed_ids = [
                out[len(in_ids):] 
                for in_ids, out in zip(mm_inputs.input_ids, generated_ids)
            ]

            output_text = self.processor.batch_decode(trimmed_ids, 
                                                skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=False)

            return output_text[0]
        except Exception as e:
            logger.error(f"生成文本时出错: {str(e)}")


    def clean(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()