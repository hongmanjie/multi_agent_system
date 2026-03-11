import argparse
import json
import os
import importlib
from fastapi import FastAPI, HTTPException
from typing import Dict, Any
import asyncio
import logging
from pydantic import BaseModel

from xt_maas.utils.logger import get_logger
from xt_maas.utils.base64_utils import base64_to_img


logger = get_logger(log_level=logging.ERROR)


class PredictionRequest(BaseModel):
    image_id: str                # 图片唯一标识符
    image: str                   # base64 编码的图片字符串
    prompt: str = "请描述这张图片的内容。"  # 可选，默认提示语

loaded_models: Dict[str, Any] = {}


def register_model_route(app: FastAPI, model_id: str, model_type: str, model_instance: Any):
    """模型注册POST /{model_id}/predict 路由
    """

    @app.post(f"/{model_type}/{model_id}/predict")
    async def predict(request: PredictionRequest):
        if model_type == "cv":
            if not hasattr(model_instance, "predict"):
                return {"error": f"模型类型 {model_type} 模型 {model_id} .没有实现 predict 方法"}
        elif model_type == "multi_modal":
            if not hasattr(model_instance, "generate"):
                return {"error": f"模型类型 {model_type} 模型 {model_id} .没有实现 generate 方法"}
        else:
            return {"error": f"未知的模型类型: {model_type}"}

        try:
            input_data = {
                "image_id": request.image_id,
                "image": request.image
            }
            image_data = base64_to_img(input_data["image"])
            if model_type == "cv":

                result = await asyncio.to_thread(model_instance.predict, image_data)
            elif model_type == "multi_modal":
                # 将image_data转为PIL Image
                from PIL import Image
                image_pil = Image.fromarray(image_data)
                result = await asyncio.to_thread(model_instance.generate, 
                                                 image_pil, 
                                                 prompt=request.prompt,
                                                 temperature=0.7,
                                                 top_p=0.9)
            else:
                result = {"error": f"未知的模型类型: {model_type}"}
            return {
                "model_id": model_id,
                "image_id": request.image_id,
                "image_width": image_data.shape[1],
                "image_height": image_data.shape[0],
                "prediction": result
            }
        
        except Exception as e:
            logger.error(f"[ERROR] 模型 {model_id} 推理失败: {e}")
            raise HTTPException(status_code=500, detail=f"模型推理失败: {str(e)}")

def load_models_from_config(app: FastAPI, config_path: str):
    """加载model_server.json
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"模型配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        model_configs = json.load(f)

    for cfg in model_configs:
        model_id = cfg.get("model_id")
        if not cfg.get("enabled", False):
            logger.info(f"[INFO] 模型 '{model_id}' 未启用，跳过加载。")
            continue

        module_path = cfg.get("module")
        class_name = cfg.get("class")
        model_type = cfg.get("model_type")
        model_params = cfg.get("model_config", {})

        logger.info(f"[INFO] 正在加载模型: {model_id} ({module_path}.{class_name})")

        try:
            module = importlib.import_module(module_path)
            model_cls = getattr(module, class_name)
            model_instance = model_cls(**model_params)
            loaded_models[model_id] = model_instance

            register_model_route(app, model_id, model_type, model_instance)
            logger.info(f"[SUCCESS] 模型 '{model_id}' 加载成功，路由已注册。")

        except Exception as e:
            logger.error(f"[ERROR] 加载模型 '{model_id}' 失败: {e}")




def add_server_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--config_path', default="configs/model_server.json", type=str, help='The target model id')
    parser.add_argument('--host', default='0.0.0.0', help='Host to listen')
    parser.add_argument('--port', type=int, default=8000, help='Server port')
    parser.add_argument('--debug', default='debug', help='Set debug level.')

def get_app(args):
    app = FastAPI(
        title='xt_maas_server',
        version='1.0',
        debug=True)
    
    load_models_from_config(app, args.config_path)

    app.state.args = args
    # app.include_router(api_router)


    # 打印所有已注册的路由
    print("\n 已注册的路由列表：")
    for route in app.routes:
        if hasattr(route, 'path') and hasattr(route, 'methods'):
            methods = ", ".join(route.methods) if route.methods else "ANY"
            print(f"  {route.path} [{methods}]")

    return app


if __name__ == '__main__':
    import uvicorn
    parser = argparse.ArgumentParser('xt_maas_server')
    add_server_args(parser)
    args = parser.parse_args()
    app = get_app(args)
    uvicorn.run(app, host=args.host, port=args.port)

