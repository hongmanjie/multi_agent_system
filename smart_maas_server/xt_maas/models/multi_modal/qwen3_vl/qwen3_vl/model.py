# 文件路径应该是: qwen3_vl_moe.py (与当前模块同目录下的 qwen3_vl_moe.py 文件)
import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import torch
from PIL import Image
import logging
from transformers import (
    AutoProcessor, 
    Qwen3VLMoeForConditionalGeneration)

from xt_maas.utils.device_utils import get_gpu_memory_usage_nvidia_smi
from xt_maas.utils.logger import get_logger

logger = get_logger()

class Qwen3VL_Model:
    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-VL-30B-A3B-Thinking",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        max_new_tokens: int = 128, 
        min_image_size: int = 256,   # 最小图像尺寸，单位为28x28块
        max_image_size: int = 512,  # 最大图像尺寸，单位为28x28块
        use_flash_attention: bool = True,
    ):
        """
        Qwen3-VL-MoE 模型初始化
        Args:
            model_path: 模型路径或名称
            torch_dtype: torch 数据类型
            device_map: 指定设备
            max_new_tokens: 最大生成长度
            use_flash_attention: 是否使用flash attention 2加速
        """
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.use_flash_attention = use_flash_attention

        self.min_pixels = min_image_size * 28 * 28
        self.max_pixels = max_image_size * 28 * 28
        self.model = None
        self.processor = None

        try:
            self._load_model()
            logger.info(f"Qwen3-VL-MoE 模型成功加载: {self.device}")
        except Exception as e:
            logger.error(f"加载 Qwen3-VL-MoE 模型失败: {str(e)}")
            raise

    def _load_model(self):
        """加载模型和处理器"""        
        # 准备模型加载参数
        model_kwargs = {
            "pretrained_model_name_or_path": self.model_path,
            "torch_dtype": self.torch_dtype,
            "device_map": None
        }
        
        # 如果启用flash attention 2
        if self.use_flash_attention and torch.cuda.is_available():
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.info("启用 Flash Attention 2 加速")

        device_map = self.device_map.lower()

        # 显卡配置 
        if device_map == "auto":
            if torch.cuda.is_available():
                gpu_info = get_gpu_memory_usage_nvidia_smi()
                logger.info(f"当前设备的 GPU 数量: {len(gpu_info)}")
                if gpu_info and len(gpu_info) > 0:
                    best_gpu = max(gpu_info, key=lambda x: x['free_memory_MB'])
                    self.device = f"cuda:{best_gpu['device_id']}"
                    logger.info(f"自动选择剩余显存最大的 GPU: {self.device}，剩余显存: {best_gpu['free_memory_MB']:.2f} MB")
                    model_kwargs["device_map"] = self.device
                else:
                    logger.warning("未能获取 GPU 信息，将使用 CPU 运行")
                    self.device = "cpu"
                    model_kwargs["device_map"] = self.device
            else:
                logger.warning("CUDA不可用，将使用CPU运行")
                self.device = "cpu"
                model_kwargs["device_map"] = self.device
        elif "," in device_map:
            # 多卡情况
            devices = [d.strip() for d in device_map.split(",")]
            logger.info(f"使用多卡模式，指定设备列表: {devices}")
            
            if not all(d.startswith("cuda:") for d in devices):
                raise ValueError(f"无效的设备列表: {device_map}。请使用 'cuda:X,cuda:Y' 格式")

            model_kwargs["device_map"] = "auto"
            self.device = "cuda"
        else: # 单卡情况
            self.device = device_map
            logger.info(f"使用单卡模式，指定设备: {device_map}")
            model_kwargs["device_map"] = self.device

        # 加载模型
        self.model = Qwen3VLMoeForConditionalGeneration.from_pretrained(**model_kwargs)
        
        # 加载处理器
        self.processor = AutoProcessor.from_pretrained(
            self.model_path,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels
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
            
            # 转换为 PIL Image
            image = Image.fromarray(image)
            
            # 构建消息
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": text if text else "请详细描述这张图片的内容"}
                ]
            }]

            # 准备输入
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.device)

            # 生成响应
            generated_ids = self.model.generate(
                **inputs, 
                max_new_tokens=self.max_new_tokens,
                temperature=temperature,
                top_p=top_p
            )

            # 修剪输入ID
            generated_ids_trimmed = [
                out_ids[len(in_ids):] 
                for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]

            # 解码输出
            output_text = self.processor.batch_decode(
                generated_ids_trimmed, 
                skip_special_tokens=True, 
                clean_up_tokenization_spaces=False
            )

            return output_text[0]
            
        except Exception as e:
            logger.error(f"生成文本时出错: {str(e)}")
            raise

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