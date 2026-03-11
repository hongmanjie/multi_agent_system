import json
import os

def load_config(config_path):
    """
    加载指定路径的 JSON 配置文件。
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"配置文件未找到: {config_path}")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"JSON 格式错误: {e.msg}", e.doc, e.pos)
    except Exception as e:
        raise Exception(f"读取配置文件时发生异常: {e}")