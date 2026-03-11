"""
人脸检测和人脸识别API测试用例
用于测试运行在8010端口的MaaS服务
"""
import pytest
import requests
import base64
import os
from typing import Dict, Any


# 服务配置
BASE_URL = "http://127.0.0.1:8010"
FACE_DETECTION_ENDPOINT = f"{BASE_URL}/cv/retinaface_detection/predict"
FACE_RECOGNITION_ENDPOINT = f"{BASE_URL}/cv/arcface_recognition/predict"


def image_to_base64(image_path: str) -> str:
    """
    将图片转为Base64编码字符串
    :param image_path: 图片路径
    :return: Base64编码的字符串
    """
    with open(image_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode('utf-8')
    return base64_data


class TestFaceDetectionAPI:
    """人脸检测API测试类"""

    @pytest.fixture
    def test_image_path(self):
        """测试图片路径"""
        image_path = '../test_datas/face_detection/baby_yangmi_nining.jpg'
        # 支持相对路径和绝对路径
        if not os.path.exists(image_path):
            # 尝试从tests目录的父目录查找
            image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'test_datas', 'face_detection', 'baby_yangmi_nining.jpg')
        return image_path

    @pytest.fixture
    def face_detection_request_data(self, test_image_path):
        """构建人脸检测请求数据"""
        b64_image = image_to_base64(test_image_path)
        return {
            'request_id': "test_face_detection_001",
            'image': b64_image,
            'nms_threshold': 0.5,
            'score_threshold': 0.9,
            'roi': []
        }

    def test_face_detection_api_connection(self):
        """测试API连接是否正常"""
        # 使用一个简单的请求测试连接
        response = requests.get(BASE_URL, timeout=5)
        assert response.status_code in [200, 404, 405], "无法连接到MaaS服务"

    def test_face_detection_basic(self, face_detection_request_data):
        """测试基本的人脸检测功能"""
        response = requests.post(FACE_DETECTION_ENDPOINT, json=face_detection_request_data, timeout=30)
        
        # 断言响应状态码
        assert response.status_code == 200, f"API请求失败，状态码: {response.status_code}"
        
        # 解析响应
        result = response.json()
        
        # 验证响应结构
        assert 'model_id' in result, "响应中缺少model_id字段"
        assert 'request_id' in result, "响应中缺少request_id字段"
        assert 'prediction' in result, "响应中缺少prediction字段"
        assert result['model_id'] == 'retinaface_detection', "model_id不匹配"
        
        # 验证预测结果结构
        prediction = result['prediction']
        assert 'detections' in prediction, "prediction中缺少detections字段"
        assert 'landmarks' in prediction, "prediction中缺少landmarks字段"
        
        # 验证检测结果
        detections = prediction['detections']
        assert isinstance(detections, list), "detections应该是列表类型"
        
        if len(detections) > 0:
            # 验证检测框格式 [x1, y1, x2, y2, score]
            for detection in detections:
                assert len(detection) >= 5, f"检测框格式不正确: {detection}"
                assert detection[0] < detection[2], "x1应该小于x2"
                assert detection[1] < detection[3], "y1应该小于y2"
        
        print(f"\n人脸检测成功！检测到 {len(detections)} 个人脸")
        print(f"请求ID: {result['request_id']}")

    def test_face_detection_with_custom_thresholds(self, face_detection_request_data):
        """测试使用自定义阈值的人脸检测"""
        # 修改阈值参数
        face_detection_request_data['nms_threshold'] = 0.3
        face_detection_request_data['score_threshold'] = 0.7
        
        response = requests.post(FACE_DETECTION_ENDPOINT, json=face_detection_request_data, timeout=30)
        
        assert response.status_code == 200, f"API请求失败，状态码: {response.status_code}"
        result = response.json()
        
        assert 'prediction' in result
        assert 'detections' in result['prediction']
        
        print(f"\n自定义阈值测试成功，检测到 {len(result['prediction']['detections'])} 个人脸")

    def test_face_detection_with_roi(self, face_detection_request_data):
        """测试带ROI区域的人脸检测"""
        # 设置ROI区域
        face_detection_request_data['roi'] = [
            [50.0, 60.0, 150.0, 200.0],  # 第一个区域
        ]
        
        response = requests.post(FACE_DETECTION_ENDPOINT, json=face_detection_request_data, timeout=30)
        
        assert response.status_code == 200, f"API请求失败，状态码: {response.status_code}"
        result = response.json()
        
        assert 'prediction' in result
        print(f"\nROI区域测试成功，检测到 {len(result['prediction']['detections'])} 个人脸")


class TestFaceRecognitionAPI:
    """人脸识别API测试类"""

    @pytest.fixture
    def test_image_path(self):
        """测试图片路径"""
        image_path = '../test_datas/face_recognition/Aaron_Peirsol/Aaron_Peirsol_0001.jpg'
        # 支持相对路径和绝对路径
        if not os.path.exists(image_path):
            image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'test_datas', 'face_recognition', 
                                     'Aaron_Peirsol', 'Aaron_Peirsol_0001.jpg')
        return image_path

    @pytest.fixture
    def face_recognition_request_data(self, test_image_path):
        """构建人脸识别请求数据"""
        b64_image = image_to_base64(test_image_path)
        return {
            'request_id': "test_face_recognition_001",
            'image': b64_image
        }

    def test_face_recognition_api_connection(self):
        """测试API连接是否正常"""
        response = requests.get(BASE_URL, timeout=5)
        assert response.status_code in [200, 404, 405], "无法连接到MaaS服务"

    def test_face_recognition_basic(self, face_recognition_request_data):
        """测试基本的人脸识别功能"""
        response = requests.post(FACE_RECOGNITION_ENDPOINT, json=face_recognition_request_data, timeout=30)
        
        # 断言响应状态码
        assert response.status_code == 200, f"API请求失败，状态码: {response.status_code}"
        
        # 解析响应
        result = response.json()
        
        # 验证响应结构
        assert 'model_id' in result, "响应中缺少model_id字段"
        assert 'request_id' in result, "响应中缺少request_id字段"
        assert 'prediction' in result, "响应中缺少prediction字段"
        assert result['model_id'] == 'arcface_recognition', "model_id不匹配"
        
        # 验证预测结果结构
        prediction = result['prediction']
        assert 'embedding' in prediction, "prediction中缺少embedding字段"
        
        # 验证embedding特征向量
        embedding = prediction['embedding']
        assert isinstance(embedding, list), "embedding应该是列表类型"
        assert len(embedding) > 0, "embedding不应该为空"
        
        # 验证embedding中的元素都是数字
        for value in embedding:
            assert isinstance(value, (int, float)), "embedding中的值应该是数字类型"
        
        print(f"\n人脸识别成功！")
        print(f"请求ID: {result['request_id']}")
        print(f"特征向量维度: {len(embedding)}")
        print(f"特征向量前5个值: {embedding[:5]}")

    def test_face_recognition_embedding_dimension(self, face_recognition_request_data):
        """测试人脸识别特征向量维度"""
        response = requests.post(FACE_RECOGNITION_ENDPOINT, json=face_recognition_request_data, timeout=30)
        
        assert response.status_code == 200
        result = response.json()
        
        embedding = result['prediction']['embedding']
        # 通常人脸识别特征向量是128、256、512等维度
        assert embedding is not None, "embedding不应该为None"
        assert len(embedding) > 0, "embedding维度应该大于0"
        
        print(f"\n特征向量维度测试通过: {len(embedding)}维")

    def test_face_recognition_different_images(self):
        """测试不同图片的人脸识别"""
        test_images = [
            '../test_datas/face_recognition/Aaron_Peirsol/Aaron_Peirsol_0001.jpg',
            '../test_datas/face_recognition/Aaron_Peirsol/Aaron_Peirsol_0002.jpg',
        ]
        
        embeddings = []
        for img_path in test_images:
            if not os.path.exists(img_path):
                img_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                       'test_datas', 'face_recognition', 
                                       'Aaron_Peirsol', os.path.basename(img_path))
            
            if not os.path.exists(img_path):
                pytest.skip(f"测试图片不存在: {img_path}")
            
            b64_image = image_to_base64(img_path)
            data = {
                'request_id': f"test_multi_image_{len(embeddings)}",
                'image': b64_image
            }
            
            response = requests.post(FACE_RECOGNITION_ENDPOINT, json=data, timeout=30)
            assert response.status_code == 200
            
            result = response.json()
            embeddings.append(result['prediction']['embedding'])
        
        # 验证所有特征向量维度一致
        if len(embeddings) > 1:
            dim = len(embeddings[0])
            for emb in embeddings[1:]:
                assert len(emb) == dim, "不同图片的特征向量维度应该一致"
            
            print(f"\n多图片测试成功，共处理 {len(embeddings)} 张图片，特征维度: {dim}")


class TestFaceAPIIntegration:
    """人脸检测和识别集成测试"""

    def test_face_detection_then_recognition(self):
        """测试先检测人脸再识别的流程"""
        # 1. 先进行人脸检测
        image_path = '../test_datas/face_detection/baby_yangmi_nining.jpg'
        if not os.path.exists(image_path):
            image_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                     'test_datas', 'face_detection', 'baby_yangmi_nining.jpg')
        
        if not os.path.exists(image_path):
            pytest.skip("测试图片不存在")
        
        b64_image = image_to_base64(image_path)
        
        # 人脸检测
        detection_data = {
            'request_id': "test_integration_detection",
            'image': b64_image,
            'nms_threshold': 0.5,
            'score_threshold': 0.9,
            'roi': []
        }
        
        detection_response = requests.post(FACE_DETECTION_ENDPOINT, json=detection_data, timeout=30)
        assert detection_response.status_code == 200
        detection_result = detection_response.json()
        detections = detection_result['prediction']['detections']
        
        assert len(detections) > 0, "应该检测到至少一个人脸"
        
        # 2. 再进行人脸识别
        recognition_data = {
            'request_id': "test_integration_recognition",
            'image': b64_image
        }
        
        recognition_response = requests.post(FACE_RECOGNITION_ENDPOINT, json=recognition_data, timeout=30)
        assert recognition_response.status_code == 200
        recognition_result = recognition_response.json()
        embedding = recognition_result['prediction']['embedding']
        
        assert len(embedding) > 0, "应该生成特征向量"
        
        print(f"\n集成测试成功！")
        print(f"检测到 {len(detections)} 个人脸")
        print(f"生成特征向量维度: {len(embedding)}")


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "-s"])
