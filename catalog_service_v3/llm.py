import requests
import logging
import os
from openai import OpenAI

# 确保日志目录存在
log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
os.makedirs(log_dir, exist_ok=True)

# 创建专门的logger
logger = logging.getLogger('llm')
logger.setLevel(logging.INFO)

# 检查logger是否已经有handler
if not logger.handlers:
    # 创建FileHandler
    file_handler = logging.FileHandler(
        filename=os.path.join(log_dir, 'llm.log'),
        encoding='utf-8'
    )
    file_handler.setLevel(logging.INFO)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)

def generate_text_by_qwen3_5(prompt: str) -> str:
    """
    调用千问3.5模型生成文本
    
    Args:
        prompt: 输入提示词
        
    Returns:
        生成的文本内容
    """
    logger.info("开始调用千问3.5模型生成文本")
    logger.info(f"输入提示词长度: {len(prompt)} 字符")
    
    try:
        api_key = "sk-49f1f7ad9fcd4ca693a47edcab023388"
        client = OpenAI(
            # 各地域的API Key不同。获取API Key：https://help.aliyun.com/zh/model-studio/get-api-key
            # 若没有配置环境变量，请用阿里云百炼API Key将下行替换为：api_key="sk-xxx",
            # 各地域的base_url不同
            api_key = api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        completion = client.chat.completions.create(
            model="qwen3.5-plus",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
        )
        
        result = completion.choices[0].message.content
        logger.info(f"生成文本长度: {len(result)} 字符")
        logger.info("千问3.5模型调用成功")
        return result
        # 如需查看完整响应，请取消下列注释
        # print(completion.model_dump_json())
    except Exception as e:
        logger.error(f"千问3.5调用失败: {e}")
        logger.error("请参考文档：https://help.aliyun.com/zh/model-studio/developer-reference/error-code")
        return ""


def generate_text_by_glm5(prompt: str) -> str:
    """
    调用GLM-5模型生成文本
    
    Args:
        prompt: 输入提示词
        
    Returns:
        生成的文本内容
    """
    logger.info("开始调用GLM-5模型生成文本")
    logger.info(f"输入提示词长度: {len(prompt)} 字符")
    
    url = "https://open.bigmodel.cn/api/paas/v4/chat/completions"
    
    payload = {
        "model": "glm-4.7",
        "messages": [
            {
                "role": "system",
                "content": "你是一个有用的AI助手。"
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        "stream": False,
        "temperature": 0.7,
        "do_sample": True,
        "response_format": { "type": "text" },
        "max_tokens": 2048
    }
    
    headers = {
        "Authorization": "Bearer 2a3fe0a715e244fea9c822cfc18a0ff3.W8WJfe2HMkzTpZg6",
        "Content-Type": "application/json"
    }
    
    try:
        logger.info("发送请求到GLM-5 API")
        response = requests.post(url, json=payload, headers=headers, timeout=120)
        logger.info(f"API响应状态码: {response.status_code}")
        response.raise_for_status()
        
        result = response.json()["choices"][0]["message"]["content"]
        logger.info(f"生成文本长度: {len(result)} 字符")
        logger.info("GLM-5模型调用成功")
        return result
    except Exception as e:
        logger.error(f"LLM调用失败: {e}")
        return ""


if __name__ == "__main__":
    # 测试代码
    test_prompt = "请简单介绍一下你自己。"
    
    # 测试GLM-5
    # print("测试GLM-5模型...")
    # result_glm5 = generate_text_by_glm5(test_prompt)
    # print("GLM-5 结果:")
    # print(result_glm5)
    
    # 测试千问3.5
    print("\n测试千问3.5模型...")
    result_qwen35 = generate_text_by_qwen3_5(test_prompt)
    print("千问3.5 结果:")
    print(result_qwen35)
