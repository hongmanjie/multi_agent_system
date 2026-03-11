import requests
import base64
import json
import os
import uuid
from pathlib import Path


def test_funasr_transcription():
    """
    测试smart_maas_server中的funasr音频转写服务
    """
    # 服务端点
    api_url = "http://127.0.0.1:8010/audio/funasr_asr/predict"
    
    # 测试音频文件路径（需要替换为实际的测试音频文件）
    test_audio_path = "/data_ssd/smart_catalog_release_v1.1/catalog_service_v3/temp_videos/task_81_1770801462_scenes_audio/task_81_1770801462_full.mp3"  # 示例路径，需要根据实际情况修改
    
    # 检查测试音频文件是否存在
    if not os.path.exists(test_audio_path):
        print(f"错误: 测试音频文件不存在: {test_audio_path}")
        print("请提供一个有效的WAV或MP3格式的测试音频文件")
        return False
    
    print(f"开始测试FunASR服务...")
    print(f"测试音频文件: {test_audio_path}")
    
    try:
        # 读取音频文件
        with open(test_audio_path, 'rb') as f:
            audio_data = f.read()
        
        # 编码为base64
        audio_data_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        # 构建请求数据
        request_id = str(uuid.uuid4())
        audio_format = Path(test_audio_path).suffix.lstrip('.').lower()
        
        payload = {
            "request_id": request_id,
            "audio_data": audio_data_base64,
            "audio_format": audio_format
        }
        
        print(f"发送请求到: {api_url}")
        print(f"请求ID: {request_id}")
        print(f"音频格式: {audio_format}")
        print(f"音频大小: {len(audio_data)} 字节")
        
        # 发送请求
        response = requests.post(
            api_url,
            json=payload,
            timeout=60  # 1分钟超时
        )
        
        # 检查响应状态
        if response.status_code == 200:
            result = response.json()
            print("\n测试成功! 响应结果:")
            print(json.dumps(result, ensure_ascii=False, indent=2))
            
            # 检查响应格式
            if 'prediction' in result:
                prediction = result['prediction']
                print("\n转写结果:")
                if 'results' in prediction:
                    for i, segment in enumerate(prediction['results']):
                        speaker = segment.get('speaker_id', 'unknown')
                        text = segment.get('text', '')
                        timestamp = segment.get('timestamp', 'unknown')
                        print(f"[{i+1}] {speaker}: {text} (时间: {timestamp})")
                else:
                    print("警告: 响应中缺少'results'字段")
            else:
                print("警告: 响应中缺少'prediction'字段")
            
            return True
        else:
            print(f"\n测试失败! 状态码: {response.status_code}")
            print(f"响应内容: {response.text}")
            return False
            
    except requests.exceptions.RequestException as e:
        print(f"\n请求异常: {str(e)}")
        print("请确保smart_maas_server服务正在运行")
        return False
    except Exception as e:
        print(f"\n测试异常: {str(e)}")
        return False


def test_funasr_endpoint_health():
    """
    测试FunASR端点是否可访问
    """
    api_url = "http://127.0.0.1:8010/audio/funasr_asr/predict"
    
    print(f"\n检查FunASR端点健康状态: {api_url}")
    
    try:
        # 发送一个简单的GET请求来检查端点是否可访问
        response = requests.get(api_url, timeout=10)
        print(f"端点响应状态: {response.status_code}")
        if response.status_code == 405:  # 预期的方法不允许，因为我们需要POST
            print("端点可访问，但需要POST请求")
            return True
        elif response.status_code == 200:
            print("端点可访问并返回200 OK")
            return True
        else:
            print(f"端点返回意外状态: {response.status_code}")
            return False
    except requests.exceptions.ConnectionError:
        print("无法连接到端点，请确保服务正在运行")
        return False
    except Exception as e:
        print(f"检查异常: {str(e)}")
        return False


if __name__ == "__main__":
    print("=====================================")
    print("FunASR服务测试")
    print("=====================================")
    
    # 检查端点健康状态
    health_ok = test_funasr_endpoint_health()
    
    if health_ok:
        print("\n进行转写测试...")
        transcribe_ok = test_funasr_transcription()
        
        if transcribe_ok:
            print("\n✅ 所有测试通过!")
        else:
            print("\n❌ 转写测试失败")
    else:
        print("\n❌ 端点健康检查失败，无法进行转写测试")
    
    print("=====================================")
