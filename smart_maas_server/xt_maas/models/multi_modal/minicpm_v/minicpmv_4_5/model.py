import numpy as np
from typing import Optional, Tuple, List, Dict, Any
import torch
from PIL import Image
import logging
from transformers import AutoModel, AutoTokenizer

from xt_maas.utils.device_utils import get_gpu_memory_usage_nvidia_smi
from xt_maas.utils.logger import get_logger

logger = get_logger()

class MiniCPM_V_Model:
    def __init__(
        self,
        model_path: str = "openbmb/MiniCPM-V-4_5",
        torch_dtype: torch.dtype = torch.bfloat16,
        device_map: str = "auto",
        max_new_tokens: int = 128,
        attn_implementation: str = "sdpa",
        enable_thinking: bool = False,
        stream: bool = True
    ):
        """
        MiniCPM-V-4_5 模型初始化
        Args:
            model_path: 模型路径或名称
            torch_dtype: torch 数据类型
            device_map: 指定设备
            max_new_tokens: 最大生成长度
            attn_implementation: 注意力机制实现方式
            enable_thinking: 是否启用思考模式
            stream: 是否使用流式输出
        """
        self.model_path = model_path
        self.torch_dtype = torch_dtype
        self.device_map = device_map
        self.max_new_tokens = max_new_tokens
        self.attn_implementation = attn_implementation
        self.enable_thinking = enable_thinking
        self.stream = stream
        
        self.model = None
        self.tokenizer = None
        self.device = None

        try:
            self._load_model()
            logger.info(f"MiniCPM-V-4_5 模型成功加载: {self.device}")
        except Exception as e:
            logger.error(f"加载 MiniCPM-V-4_5 模型失败: {str(e)}")
            raise

    def _load_model(self):
        """加载模型和tokenizer"""        
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
                else:
                    logger.warning("未能获取 GPU 信息，将使用 CPU 运行")
                    self.device = "cpu"
            else:
                logger.warning("CUDA不可用，将使用CPU运行")
                self.device = "cpu"
        elif "," in device_map:
            # 多卡情况
            devices = [d.strip() for d in device_map.split(",")]
            logger.info(f"使用多卡模式，指定设备列表: {devices}")
            if not all(d.startswith("cuda:") for d in devices):
                raise ValueError(f"无效的设备列表: {device_map}。请使用 'cuda:X,cuda:Y' 格式")
            self.device = "cuda"
        else: # 单卡情况
            self.device = device_map
            logger.info(f"使用单卡模式，指定设备: {device_map}")

        # 加载模型和tokenizer
        self.model = AutoModel.from_pretrained(
            self.model_path,
            trust_remote_code=True,
            attn_implementation=self.attn_implementation,
            torch_dtype=self.torch_dtype
        ).eval().to(self.device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            trust_remote_code=True
        )

    def generate(
        self,
        image,
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
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("模型未加载，请先初始化模型")

        try:
            # 检查输入的image格式
            if image is None:
                raise ValueError("图像输入为空")
            
            # 转换为 PIL Image
#           image = Image.fromarray(image)
            
            # 构建消息格式
            msgs = [{'role': 'user', 'content': [image, text if text else "请详细描述这张图片的内容"]}]
            
            # 调用模型的chat方法
            answer = self.model.chat(
                msgs=msgs,
                tokenizer=self.tokenizer,
                enable_thinking=self.enable_thinking,
                stream=self.stream
            )
            
            # 处理流式输出或直接输出
            if self.stream:
                generated_text = ""
                for new_text in answer:
                    generated_text += new_text
                return generated_text
            else:
                return answer
                
        except Exception as e:
            logger.error(f"生成文本时出错: {str(e)}")
            raise

    def clean(self):
        """清理资源"""
        if self.model is not None:
            del self.model
            self.model = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()