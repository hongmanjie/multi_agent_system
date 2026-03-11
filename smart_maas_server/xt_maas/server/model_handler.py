from typing import Type, Any, Dict, Generic, TypeVar, Optional, List
import asyncio
from fastapi import HTTPException, FastAPI
from pydantic import BaseModel
from PIL import Image

from xt_maas.utils.base64_utils import base64_to_img
from xt_maas.utils.logger import get_logger

logger = get_logger()

"""
处理不同模型的输入输出参数
"""
T = TypeVar('T', bound=BaseModel)

class BasePredictionRequest(BaseModel):
    request_id: str 

class CVPredictionRequest(BasePredictionRequest):
    image: str  # base64
    categories: Optional[List[str]] = None
    nms_threshold: Optional[float] = None
    score_threshold: Optional[float] = None
    roi: Optional[List[List[float]]] = None
    model_config = {
        "extra": "allow"  # 允许传入其他自定义字段
    }

class MultiModalPredictionRequest(BasePredictionRequest):
    image: Optional[str] =  None     # base64
    text: Optional[str] = None     # 文本提示词

    temperature: Optional[float] = None    
    top_p: Optional[float] = None  

class NLPPredictionRequest(BasePredictionRequest):
    prompt: str                
    temperature: Optional[float] = None    
    top_p: Optional[float] = None        
    top_k: Optional[int] = None           
    repetition_penalty: Optional[float] = None  
    max_new_tokens: Optional[int] = None     
    model_config = {
        "extra": "allow",  # 允许传入额外的字段
    }

class AudioPredictionRequest(BasePredictionRequest):
    audio_data: str  # base64编码的音频数据
    audio_format: Optional[str] = "wav"  # 音频格式
    model_config = {
        "extra": "allow"  # 允许传入其他自定义字段
    }

class BaseModelHandler(Generic[T]):
    """所有模型类型 Handler 的基类"""
    @classmethod
    def get_route_path(cls, model_id: str) -> str:
        """获取路由路径"""
        raise NotImplementedError

    @classmethod
    def get_request_model(cls) -> Type[T]:
        """获取请求模型类"""
        raise NotImplementedError

    @classmethod
    def get_method_name(cls) -> str:
        """获取要调用的模型方法名"""
        raise NotImplementedError

    @classmethod
    def build_response(
        cls,
        model_id: str,
        request_id: str,
        image_data: Any = None,  # 可能为 None，如 NLP 模型无图像
        result: Any = None,
        ** extra
    ) -> Dict[str, Any]:
        """构建响应结果"""
        raise NotImplementedError

    @classmethod
    def extract_input(cls, request: T, model_instance: Any, image_data: Any) -> Dict[str, Any]:
        """从请求中提取模型输入参数"""
        raise NotImplementedError

    @classmethod
    async def handle_request(cls, request: T, model_instance: Any, model_id: str):
        """处理请求的主流程"""
        try:
            # 验证请求模型类型
            RequestModel = cls.get_request_model()
            if not isinstance(request, RequestModel):
                raise HTTPException(status_code=400, detail="无效的请求模型类型")

            # 获取并验证模型方法
            method_name = cls.get_method_name()
            if not hasattr(model_instance, method_name):
                raise HTTPException(
                    status_code=500, 
                    detail=f"模型未实现 '{method_name}' 方法"
                )

            # 提取图像数据（仅对需要的类型）
            image_data = None
            if hasattr(request, "image") and request.image:
                try:
                    image_data = base64_to_img(request.image)
                except Exception as e:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"图像解码失败: {str(e)}"
                    )

            input_kwargs = cls.extract_input(request, model_instance, image_data)
            # 在独立线程中运行模型推理
            method = getattr(model_instance, method_name)
            result = await asyncio.to_thread(method, **input_kwargs)

            # 构建返回结果
            extra_params = {}
            if image_data is not None:
                extra_params = {
                    "image_width": image_data.size[0],
                    "image_height": image_data.size[1]
                }

            return cls.build_response(
                model_id=model_id,
                request_id=request.request_id,
                image_data=image_data,
                result=result,
                ** extra_params
            )

        except HTTPException:
            # 已处理的HTTP异常直接抛出
            raise
        except Exception as e:
            logger.error(f"[ERROR] 模型推理失败: {e}", exc_info=True)
            raise HTTPException(
                status_code=500, 
                detail=f"推理失败: {str(e)}"
            )

    @classmethod
    def register_route(cls, app: FastAPI, model_id: str, model_instance: Any):
        """为模型注册路由"""
        route_path = cls.get_route_path(model_id)
        request_model = cls.get_request_model()

        @app.post(route_path, description=f"处理{model_id}模型的请求")
        async def endpoint(request: request_model):            
            return await cls.handle_request(request, model_instance, model_id)
        
        logger.info(f"[SUCCESS] 模型 '{model_id}' 加载成功，路由 '{route_path}' 已注册。")


class CVModelHandler(BaseModelHandler[CVPredictionRequest]):
    # 推理时支持除image 以外其他参数的模型类名列表
    DETECT_CLASS_NAMES = ["RetinaFaceDetection", "YOLOv12_ONNX_PERSON", "YOLOv12_ONNX_HELMET", "YOLOv12_ONNX_POSE", "DAMOYOLO_ONNX_HELMET", "RexOmniModel", "SAM3Model"]

    @classmethod
    def get_route_path(cls, model_id: str) -> str:
        return f"/cv/{model_id}/predict"

    @classmethod
    def get_request_model(cls) -> Type[CVPredictionRequest]:
        return CVPredictionRequest

    @classmethod
    def get_method_name(cls) -> str:
        return "predict"

    @classmethod
    def extract_input(cls, request: CVPredictionRequest, model_instance: Any, image_data: Any) -> Dict[str, Any]:
        input_kwargs = {"image": image_data}
        #  传入模型的参数排除Base request中的值比如request_id
        filter_list = set(BasePredictionRequest.model_fields.keys())
        exclude_fields = filter_list.union({'image'})
        model_class_name = model_instance.__class__.__name__
        if model_class_name in cls.DETECT_CLASS_NAMES:
            request_kwargs = request.model_dump(exclude_none=True, exclude=exclude_fields)
            input_kwargs.update(request_kwargs)

        return input_kwargs

    @classmethod
    def build_response(
        cls, 
        model_id: str, 
        request_id: str, 
        result=None, 
        **extra
    ) -> Dict[str, Any]:
        return {
            "model_id": model_id,
            "request_id": request_id,
            "image_width": extra.get("image_width"),
            "image_height": extra.get("image_height"),
            "prediction": result
        }


class MultiModalModelHandler(BaseModelHandler[MultiModalPredictionRequest]):
    """多模态模型处理器"""
    # 推理时支持除image，text 以外其他参数的模型类名列表
    LVLM_CLASS_NAMES = ["Qwen2_5VL_Model"]

    @classmethod
    def get_route_path(cls, model_id: str) -> str:
        return f"/multi_modal/{model_id}/predict"

    @classmethod
    def get_request_model(cls) -> Type[MultiModalPredictionRequest]:
        return MultiModalPredictionRequest

    @classmethod
    def get_method_name(cls) -> str:
        return "generate"

    @classmethod
    def extract_input(
        cls, 
        request: MultiModalPredictionRequest, 
        model_instance: Any,
        image_data: Any
    ) -> Dict[str, Any]:    
        input_kwargs = {"image": image_data,
                        "text": request.text}
        filter_list = set(BasePredictionRequest.model_fields.keys())
        exclude_fields = filter_list.union({'image','text'})
        model_class_name = model_instance.__class__.__name__
        if model_class_name in cls.LVLM_CLASS_NAMES:
            request_kwargs = request.model_dump(exclude_none=True, exclude=exclude_fields)
            input_kwargs.update(request_kwargs)

        return input_kwargs


    @classmethod
    def build_response(
        cls, 
        model_id: str, 
        request_id: str, 
        result=None, 
        **extra
    ) -> Dict[str, Any]:
        return {
            "model_id": model_id,
            "request_id": request_id,
            "image_width": extra.get("image_width"),
            "image_height": extra.get("image_height"),
            "prediction": result
        }


class NLPModelHandler(BaseModelHandler[NLPPredictionRequest]):
    """自然语言处理模型处理器"""
    @classmethod
    def get_route_path(cls, model_id: str) -> str:
        return f"/nlp/{model_id}/predict"

    @classmethod
    def get_request_model(cls) -> Type[NLPPredictionRequest]:
        return NLPPredictionRequest

    @classmethod
    def get_method_name(cls) -> str:
        return "generate"  

    @classmethod
    def extract_input(
        cls, 
        request: NLPPredictionRequest, 
        model_instance: Any,
        image_data: Any
    ) -> Dict[str, Any]:

        input_kwargs = {}
        #  传入模型的参数排除Base request中的值比如request_id
        exclude_fields = set(BasePredictionRequest.model_fields.keys())
        request_kwargs = request.model_dump(exclude_none=True, exclude=exclude_fields)
        input_kwargs.update(request_kwargs)

        # model_class_name = model_instance.__class__.__name__
        # for k, v in input_kwargs.items():
        #     print(f"=={model_class_name}:==key:{k}")

        return input_kwargs

    @classmethod
    def build_response(
        cls, 
        model_id: str, 
        request_id: str, 
        result=None, 
        **extra
    ) -> Dict[str, Any]:
        return {
            "model_id": model_id,
            "request_id": request_id,
            "prediction": result
        }


class AudioModelHandler(BaseModelHandler[AudioPredictionRequest]):
    """音频模型处理器"""
    @classmethod
    def get_route_path(cls, model_id: str) -> str:
        return f"/audio/{model_id}/predict"

    @classmethod
    def get_request_model(cls) -> Type[AudioPredictionRequest]:
        return AudioPredictionRequest

    @classmethod
    def get_method_name(cls) -> str:
        return "predict"

    @classmethod
    def extract_input(
        cls, 
        request: AudioPredictionRequest, 
        model_instance: Any,
        image_data: Any
    ) -> Dict[str, Any]:
        import base64
        
        # 解码base64音频数据
        try:
            audio_data = base64.b64decode(request.audio_data)
        except Exception as e:
            raise HTTPException(
                status_code=400, 
                detail=f"音频数据解码失败: {str(e)}"
            )
        
        input_kwargs = {
            "audio_data": audio_data,
            "audio_format": request.audio_format
        }
        
        # 传入其他参数
        filter_list = set(BasePredictionRequest.model_fields.keys())
        exclude_fields = filter_list.union({'audio_data', 'audio_format'})
        request_kwargs = request.model_dump(exclude_none=True, exclude=exclude_fields)
        input_kwargs.update(request_kwargs)

        return input_kwargs

    @classmethod
    def build_response(
        cls, 
        model_id: str, 
        request_id: str, 
        result=None, 
        **extra
    ) -> Dict[str, Any]:
        return {
            "model_id": model_id,
            "request_id": request_id,
            "prediction": result
        }


# 模型类型与处理器的映射，新增了nlp类型
MODEL_HANDLERS = {
    "cv": CVModelHandler,
    "multi_modal": MultiModalModelHandler,
    "nlp": NLPModelHandler,
    "audio": AudioModelHandler
}

def register_model(app: FastAPI, model_type: str, model_id: str, model_instance: Any):
    """根据模型类型注册对应的路由和处理器"""
    if model_type not in MODEL_HANDLERS:
        raise ValueError(f"不支持的模型类型: {model_type},未找到对应的处理器")

    handler = MODEL_HANDLERS[model_type]
    handler.register_route(app, model_id, model_instance)
