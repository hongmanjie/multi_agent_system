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
from xt_maas.server.model_handler import register_model

logger = get_logger()

loaded_models: Dict[str, Any] = {}

def load_models_from_config(app: FastAPI, config_path: str):
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

        logger.info(f"[INFO] 正在加载模型: {model_id} ({module_path}.{class_name})，类型：{model_type}")

        try:
            module = importlib.import_module(module_path)
            model_cls = getattr(module, class_name)
            model_instance = model_cls(**model_params)
            register_model(app, model_type, model_id, model_instance)

        except Exception as e:
            logger.error(f"[ERROR] 加载模型 '{model_id}' 失败: {e}")
    

def add_server_args(parser: argparse.ArgumentParser):
    parser.add_argument(
        '--config_path', default="configs/model_server.json", type=str, help='The target model id')
    parser.add_argument('--host', default='0.0.0.0', help='Host to listen')
    parser.add_argument('--port', type=int, default=8010, help='Server port')
    parser.add_argument('--debug', default='debug', help='Set debug level.')

def get_app(args):
    app = FastAPI(
        title='xt_maas_server',
        version='1.0',
        debug=True)
    
    load_models_from_config(app, args.config_path)

    app.state.args = args

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
