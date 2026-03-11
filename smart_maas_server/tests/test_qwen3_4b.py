#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Qwen3:4B 模型专用测试脚本
该脚本通过 MAAS 服务 API 测试 Qwen3:4B 模型的功能
"""

import requests
import json
import time


def test_qwen3_4b_api():
    """
    通过 MAAS 服务 API 测试 Qwen3:4B 模型
    """
    # 服务地址
    BASE_URL = "http://127.0.0.1:8010"
    model_id = "qwen3_4b"
    model_type = "nlp"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"
    
    # 测试用例
    test_cases = [
        {
            "prompt": "请简单介绍一下人工智能的发展历史。",
            "temperature": 0.7,
            "max_new_tokens": 256
        },
        {
            "prompt": "什么是机器学习？请用通俗易懂的语言解释。",
            "temperature": 0.7,
            "max_new_tokens": 256
        },
        {
            "prompt": "请列举几个主要的深度学习框架并简要说明其特点。",
            "temperature": 0.8,
            "max_new_tokens": 512
        }
    ]
    
    print(f"开始测试 Qwen3:4B 模型 (model_id: {model_id}) 通过 MAAS API...")
    print(f"API 地址: {url}")
    print("-" * 60)
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n测试用例 {i}:")
        print(f"问题: {case['prompt']}")
        
        # 构造请求数据
        data = {
            'request_id': f"qwen3_4b_test_{int(time.time())}_{i}",
            'prompt': case['prompt'],
            'temperature': case['temperature'],
            'max_new_tokens': case['max_new_tokens'],
            'top_p': 0.9,
            'top_k': 50,
            'repetition_penalty': 1.05
        }
        
        try:
            # 发送请求
            start_time = time.time()
            response = requests.post(url, json=data, timeout=120)
            end_time = time.time()
            
            # 检查响应
            if response.status_code == 200:
                result = response.json()
                print(f"状态: 成功")
                print(f"响应时间: {end_time - start_time:.2f} 秒")
                
                if 'prediction' in result:
                    prediction = result['prediction']
                    if isinstance(prediction, dict):
                        # 如果返回的是字典格式，通常包含 thinking 和 gen_text
                        if 'gen_text' in prediction:
                            print(f"回答: {prediction['gen_text']}")
                        else:
                            print(f"回答: {prediction}")
                    else:
                        print(f"回答: {prediction}")
                else:
                    print(f"响应: {result}")
            else:
                print(f"状态: 失败 (HTTP {response.status_code})")
                print(f"错误信息: {response.text}")
                
        except requests.exceptions.Timeout:
            print("状态: 请求超时")
        except requests.exceptions.ConnectionError:
            print("状态: 连接错误 - 请确认 MAAS 服务已启动")
        except Exception as e:
            print(f"状态: 异常 - {str(e)}")
        
        print("-" * 60)
        time.sleep(1)  # 稍微等待一下再进行下一个测试


def test_qwen3_4b_with_varied_params():
    """
    使用不同参数测试 Qwen3:4B 模型
    """
    BASE_URL = "http://127.0.0.1:8010"
    model_id = "qwen3_4b"
    model_type = "nlp"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"
    
    # 测试不同参数组合
    test_params = [
        {
            "prompt": "请用一句话概括什么是大模型。",
            "temperature": 0.1,  # 低温度，更确定性的输出
            "max_new_tokens": 128
        },
        {
            "prompt": "请用一句话概括什么是大模型。",
            "temperature": 0.9,  # 高温度，更多样化的输出
            "max_new_tokens": 128
        }
    ]
    
    print(f"\n测试 Qwen3:4B 模型不同参数表现...")
    print("-" * 60)
    
    for i, params in enumerate(test_params, 1):
        print(f"\n参数测试 {i} (temperature={params['temperature']}):")
        
        data = {
            'request_id': f"qwen3_4b_param_test_{int(time.time())}_{i}",
            'prompt': params['prompt'],
            'temperature': params['temperature'],
            'max_new_tokens': params['max_new_tokens']
        }
        
        try:
            response = requests.post(url, json=data, timeout=60)
            
            if response.status_code == 200:
                result = response.json()
                if 'prediction' in result:
                    prediction = result['prediction']
                    if isinstance(prediction, dict) and 'gen_text' in prediction:
                        print(f"回答: {prediction['gen_text']}")
                    else:
                        print(f"回答: {prediction}")
                else:
                    print(f"响应: {result}")
            else:
                print(f"失败: HTTP {response.status_code}")
                
        except Exception as e:
            print(f"异常: {str(e)}")


if __name__ == "__main__":
    print("Qwen3:4B 模型 MAAS API 专用测试脚本")
    print("=" * 60)
    
    # 基础功能测试
    test_qwen3_4b_api()
    
    # 参数变化测试
    test_qwen3_4b_with_varied_params()
    
    print("\n测试完成！")
    print("\n注意事项:")
    print("- 请确保 MAAS 服务已启动且 Qwen3:4B 模型已正确配置")
    print("- API 地址默认为 http://127.0.0.1:8010")
    print("- 如需修改服务地址，请调整 BASE_URL 变量")