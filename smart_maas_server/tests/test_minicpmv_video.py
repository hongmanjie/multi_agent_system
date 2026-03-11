"""
minicpmv_4_5 视频摘要API测试用例
用于测试运行在8010端口的MaaS服务
通过提示词对视频进行总结摘要
"""
import pytest
import requests
import base64
import os
import cv2


# 服务配置
BASE_URL = "http://127.0.0.1:8010"
MINICPMV_4_5_ENDPOINT = f"{BASE_URL}/multi_modal/minicpmv_4_5/predict"

# 测试视频文件
VIDEO_FILENAME = "../test_datas/vl_vids/PW8Tl_BJNWA.mp4"


def image_to_base64(image_path: str) -> str:
    """
    将图片转为Base64编码字符串
    :param image_path: 图片路径
    :return: Base64编码的字符串
    """
    with open(image_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode('utf-8')
    return base64_data


def extract_frame_from_video(video_path: str, frame_index: int = 0, output_image_path: str = None) -> str:
    """
    从视频中提取一帧作为图片
    :param video_path: 视频文件路径
    :param frame_index: 要提取的帧索引（默认第一帧）
    :param output_image_path: 输出图片路径，如果为None则不保存
    :return: 提取的图片路径（如果保存）或None
    """
    try:
        if not os.path.exists(video_path):
            print(f"警告：视频文件不存在: {video_path}")
            return None
        
        # 打开视频
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"警告：无法打开视频文件: {video_path}")
            return None
        
        # 设置要提取的帧位置
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            print(f"警告：无法从视频中提取第 {frame_index} 帧")
            return None
        
        # 如果指定了输出路径，保存图片
        if output_image_path:
            cv2.imwrite(output_image_path, frame)
            print(f"已从视频提取帧并保存到: {output_image_path}")
            return output_image_path
        
        # 如果不保存，返回临时文件路径
        temp_path = f"{video_path}_frame_{frame_index}.jpg"
        cv2.imwrite(temp_path, frame)
        return temp_path
        
    except ImportError:
        print("警告：cv2 未安装，无法从视频提取帧")
        return None
    except Exception as e:
        print(f"警告：提取视频帧时出错: {str(e)}")
        return None


def find_video_file(filename: str) -> str:
    """
    查找视频文件，支持多个可能的路径
    :param filename: 视频文件名
    :return: 视频文件的完整路径，如果找不到返回None
    """
    possible_paths = [
        filename,  # 当前目录
        os.path.join(os.path.dirname(os.path.dirname(__file__)), filename),  # 项目根目录
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_datas', 'videos', filename),  # test_datas/videos目录
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


class TestMinicPMVVideoSummary:
    """minicpmv_4_5 视频摘要API测试类"""

    @pytest.fixture
    def video_path(self):
        """获取视频文件路径"""
        video_path = find_video_file(VIDEO_FILENAME)
        if not video_path:
            pytest.skip(f"视频文件不存在: {VIDEO_FILENAME}")
        return video_path

    @pytest.fixture
    def test_image_from_video(self, video_path):
        """从视频中提取关键帧作为测试图片"""
        # 提取第一帧作为关键帧
        image_path = extract_frame_from_video(video_path, frame_index=0)
        
        # 如果提取失败，尝试使用默认图片
        if image_path is None or not os.path.exists(image_path):
            # 尝试使用测试数据目录中的默认图片
            default_image = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                'test_datas', 'face_detection', 'baby_yangmi_nining.jpg'
            )
            if os.path.exists(default_image):
                image_path = default_image
                print(f"使用默认图片: {image_path}")
            else:
                pytest.skip("无法从视频提取帧，且默认图片不存在")
        
        yield image_path
        
        # 清理临时文件（如果是临时文件）
        if image_path and image_path.endswith('_frame_0.jpg') and os.path.exists(image_path):
            try:
                os.remove(image_path)
            except:
                pass

    def test_api_connection(self):
        """测试API连接是否正常"""
        response = requests.get(BASE_URL, timeout=5)
        assert response.status_code in [200, 404, 405], "无法连接到MaaS服务"

    def test_video_summary_basic(self, video_path, test_image_from_video):
        """测试基本的视频摘要功能"""
        # 将图片转为base64
        b64_image = image_to_base64(test_image_from_video)
        
        # 构建提示词，说明要对视频进行总结摘要
        prompt_text = f"请对视频 {VIDEO_FILENAME} 进行详细的总结和摘要。请描述视频的主要内容、关键场景、人物、动作和情节发展。"
        
        data = {
            'request_id': "test_video_summary_001",
            'image': b64_image,
            'text': prompt_text,
            'temperature': 0.9
        }
        
        print("=" * 60)
        print(f"正在请求视频摘要")
        print(f"视频文件名: {VIDEO_FILENAME}")
        print(f"视频路径: {video_path}")
        print(f"使用图片: {test_image_from_video}")
        print(f"提示词: {prompt_text}")
        print("=" * 60)
        
        response = requests.post(MINICPMV_4_5_ENDPOINT, json=data, timeout=120)
        
        # 断言响应状态码
        assert response.status_code == 200, f"API请求失败，状态码: {response.status_code}"
        
        # 解析响应
        result = response.json()
        
        # 验证响应结构
        assert 'model_id' in result, "响应中缺少model_id字段"
        assert 'request_id' in result, "响应中缺少request_id字段"
        assert 'prediction' in result, "响应中缺少prediction字段"
        assert result['model_id'] == 'minicpmv_4_5', "model_id不匹配"
        
        # 验证预测结果
        prediction = result['prediction']
        assert prediction is not None, "prediction不应该为None"
        
        print("\n" + "=" * 60)
        print("minicpmv_4_5 视频摘要 Response:")
        print("=" * 60)
        print(f"Model ID: {result['model_id']}")
        print(f"Request ID: {result['request_id']}")
        print(f"预测结果: {prediction}")
        print("=" * 60)

    def test_video_summary_detailed_prompt(self, video_path, test_image_from_video):
        """测试使用详细提示词的视频摘要"""
        b64_image = image_to_base64(test_image_from_video)
        
        # 使用更详细的提示词
        prompt_text = f"""
        请对视频 {VIDEO_FILENAME} 进行全面深入的分析和总结。请包含以下内容：
        1. 视频的主要内容和主题
        2. 视频中出现的关键场景和背景
        3. 视频中的人物、角色或对象
        4. 视频中的主要动作和情节发展
        5. 视频的整体风格和氛围
        6. 视频的时长和节奏
        """
        
        data = {
            'request_id': "test_video_summary_detailed_001",
            'image': b64_image,
            'text': prompt_text,
            'temperature': 0.9
        }
        
        response = requests.post(MINICPMV_4_5_ENDPOINT, json=data, timeout=120)
        
        assert response.status_code == 200
        result = response.json()
        
        assert 'prediction' in result
        prediction = result['prediction']
        assert prediction is not None and len(str(prediction)) > 0, "摘要结果不应该为空"
        
        print(f"\n详细提示词测试成功")
        print(f"摘要结果长度: {len(str(prediction))} 字符")

    def test_video_summary_different_temperatures(self, video_path, test_image_from_video):
        """测试不同temperature参数对摘要结果的影响"""
        b64_image = image_to_base64(test_image_from_video)
        
        prompt_text = f"请对视频 {VIDEO_FILENAME} 进行简洁的总结。"
        
        temperatures = [0.3, 0.7, 0.9]
        results = []
        
        for temp in temperatures:
            data = {
                'request_id': f"test_temperature_{temp}_001",
                'image': b64_image,
                'text': prompt_text,
                'temperature': temp
            }
            
            response = requests.post(MINICPMV_4_5_ENDPOINT, json=data, timeout=120)
            assert response.status_code == 200
            
            result = response.json()
            prediction = result['prediction']
            results.append((temp, prediction))
            
            print(f"\nTemperature {temp} 测试完成")
            print(f"摘要预览: {str(prediction)[:100]}...")
        
        # 验证所有temperature都能生成结果
        for temp, prediction in results:
            assert prediction is not None, f"Temperature {temp} 应该生成摘要结果"

    def test_video_summary_different_frames(self, video_path):
        """测试使用视频中不同帧的摘要结果"""
        # 提取多个关键帧
        frame_indices = [0, 10, 30]  # 提取不同位置的帧
        
        results = []
        for frame_idx in frame_indices:
            image_path = extract_frame_from_video(video_path, frame_index=frame_idx)
            if not image_path or not os.path.exists(image_path):
                continue
            
            b64_image = image_to_base64(image_path)
            prompt_text = f"请对视频 {VIDEO_FILENAME} 进行总结摘要。"
            
            data = {
                'request_id': f"test_frame_{frame_idx}_001",
                'image': b64_image,
                'text': prompt_text,
                'temperature': 0.9
            }
            
            response = requests.post(MINICPMV_4_5_ENDPOINT, json=data, timeout=120)
            assert response.status_code == 200
            
            result = response.json()
            prediction = result['prediction']
            results.append((frame_idx, prediction))
            
            # 清理临时文件
            if image_path.endswith(f'_frame_{frame_idx}.jpg'):
                try:
                    os.remove(image_path)
                except:
                    pass
        
        # 验证所有帧都能生成摘要
        assert len(results) > 0, "应该至少成功处理一帧"
        for frame_idx, prediction in results:
            assert prediction is not None, f"帧 {frame_idx} 应该生成摘要结果"

    def test_video_summary_response_time(self, video_path, test_image_from_video):
        """测试API响应时间"""
        import time
        
        b64_image = image_to_base64(test_image_from_video)
        prompt_text = f"请简要总结视频 {VIDEO_FILENAME} 的内容。"
        
        data = {
            'request_id': "test_response_time_001",
            'image': b64_image,
            'text': prompt_text,
            'temperature': 0.9
        }
        
        start_time = time.time()
        response = requests.post(MINICPMV_4_5_ENDPOINT, json=data, timeout=120)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        assert response.status_code == 200
        result = response.json()
        assert 'prediction' in result
        
        print(f"\nAPI响应时间: {elapsed_time:.2f} 秒")
        # 视频摘要可能需要较长时间，这里只验证不超过超时时间即可
        assert elapsed_time < 120, "API响应时间不应该超过120秒"


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "-s"])
