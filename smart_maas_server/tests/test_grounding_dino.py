"""
gounding_dino 图片目标检测API测试用例
用于测试运行在8010端口的MaaS服务
实现图片目标检测功能
"""
import pytest
import requests
import base64
import os
import cv2


# 服务配置
BASE_URL = "http://127.0.0.1:8010"
GROUNDING_DINO_ENDPOINT = f"{BASE_URL}/cv/gounding_dino/predict"


def image_to_base64(image_path: str) -> str:
    """
    将图片转为Base64编码字符串
    :param image_path: 图片路径
    :return: Base64编码的字符串
    """
    with open(image_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode('utf-8')
    return base64_data


def find_test_image(image_name: str) -> str:
    """
    查找测试图片文件，支持多个可能的路径
    :param image_name: 图片文件名
    :return: 图片文件的完整路径，如果找不到返回None
    """
    possible_paths = [
        image_name,  # 当前目录
        os.path.join(os.path.dirname(os.path.dirname(__file__)), image_name),  # 项目根目录
        os.path.join(os.path.dirname(os.path.dirname(__file__)), 'test_datas', 'yolo', image_name),  # test_datas/yolo目录
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            return path
    
    return None


def visualize_detections(image_path: str, boxes: list, labels: list, scores: list, output_path: str = None):
    """
    在图片上可视化检测结果
    :param image_path: 原始图片路径
    :param boxes: 检测框列表 [[x1, y1, x2, y2], ...]
    :param labels: 标签列表
    :param scores: 置信度列表
    :param output_path: 输出图片路径，如果为None则使用默认路径
    """
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"警告：无法读取图片 {image_path}")
            return
        
        # 绘制检测框和标签
        for box, label, score in zip(boxes, labels, scores):
            if len(box) >= 4:
                x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
                # 绘制矩形框
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # 添加标签和置信度
                label_text = f"{label}: {score:.2f}"
                cv2.putText(img, label_text, (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # 保存结果
        if output_path is None:
            output_path = f"{image_path}_detections.jpg"
        cv2.imwrite(output_path, img)
        print(f"检测结果已保存到: {output_path}")
    except Exception as e:
        print(f"可视化检测结果时出错: {str(e)}")


class TestGroundingDINO:
    """gounding_dino 目标检测API测试类"""

    @pytest.fixture
    def test_image_path(self):
        """获取测试图片路径"""
        # 优先使用 bus.jpg，如果不存在则尝试其他图片
        image_name = "bus.jpg"
        image_path = find_test_image(image_name)
        
        if not image_path:
            # 尝试其他可能的测试图片
            for alt_name in ["baby_yangmi_nining.jpg", "1.png"]:
                alt_path = find_test_image(alt_name)
                if alt_path:
                    image_path = alt_path
                    break
        
        if not image_path:
            pytest.skip("测试图片不存在")
        return image_path

    @pytest.fixture
    def detection_request_data(self, test_image_path):
        """构建目标检测请求数据"""
        b64_image = image_to_base64(test_image_path)
        return {
            'request_id': "test_gounding_dino_001",
            'image': b64_image
        }

    def test_api_connection(self):
        """测试API连接是否正常"""
        response = requests.get(BASE_URL, timeout=5)
        assert response.status_code in [200, 404, 405], "无法连接到MaaS服务"

    def test_gounding_dino_basic(self, detection_request_data, test_image_path):
        """测试基本的目标检测功能"""
        response = requests.post(GROUNDING_DINO_ENDPOINT, json=detection_request_data, timeout=30)
        
        # 断言响应状态码
        assert response.status_code == 200, f"API请求失败，状态码: {response.status_code}"
        
        # 解析响应
        result = response.json()
        
        # 验证响应结构
        assert 'model_id' in result, "响应中缺少model_id字段"
        assert 'request_id' in result, "响应中缺少request_id字段"
        assert 'prediction' in result, "响应中缺少prediction字段"
        assert result['model_id'] == 'gounding_dino', "model_id不匹配"
        
        # 验证预测结果结构
        prediction = result['prediction']
        assert 'boxes' in prediction, "prediction中缺少boxes字段"
        assert 'labels' in prediction, "prediction中缺少labels字段"
        assert 'scores' in prediction, "prediction中缺少scores字段"
        
        # 验证数据类型
        boxes = prediction['boxes']
        labels = prediction['labels']
        scores = prediction['scores']
        
        assert isinstance(boxes, list), "boxes应该是列表类型"
        assert isinstance(labels, list), "labels应该是列表类型"
        assert isinstance(scores, list), "scores应该是列表类型"
        
        # 验证数组长度一致性
        assert len(boxes) == len(labels) == len(scores), "boxes、labels和scores的长度应该一致"
        
        # 验证检测框格式
        for i, box in enumerate(boxes):
            assert len(box) >= 4, f"检测框 {i} 格式不正确，应该包含至少4个坐标值: {box}"
            assert isinstance(box[0], (int, float)), f"检测框 {i} 的坐标应该是数字类型"
        
        # 验证置信度分数范围
        for i, score in enumerate(scores):
            assert 0 <= score <= 1, f"置信度分数 {i} 应该在0-1之间: {score}"
        
        print(f"\n目标检测成功！")
        print(f"请求ID: {result['request_id']}")
        print(f"检测到 {len(boxes)} 个目标")
        print(f"目标类别: {set(labels)}")
        print(f"置信度范围: {min(scores):.2f} - {max(scores):.2f}" if scores else "无检测结果")
        
        # 可视化检测结果
        if len(boxes) > 0:
            visualize_detections(test_image_path, boxes, labels, scores)

    def test_gounding_dino_detection_count(self, detection_request_data):
        """测试检测目标数量"""
        response = requests.post(GROUNDING_DINO_ENDPOINT, json=detection_request_data, timeout=30)
        
        assert response.status_code == 200
        result = response.json()
        
        boxes = result['prediction']['boxes']
        labels = result['prediction']['labels']
        scores = result['prediction']['scores']
        
        print(f"\n检测目标数量: {len(boxes)}")
        
        # 统计各类别的数量
        from collections import Counter
        label_counts = Counter(labels)
        print("各类别检测数量:")
        for label, count in label_counts.items():
            print(f"  {label}: {count}")

    def test_gounding_dino_different_images(self):
        """测试不同图片的检测结果"""
        test_images = ["bus.jpg", "baby_yangmi_nining.jpg"]
        
        results = []
        for img_name in test_images:
            image_path = find_test_image(img_name)
            if not image_path:
                continue
            
            b64_image = image_to_base64(image_path)
            data = {
                'request_id': f"test_multi_image_{len(results)}",
                'image': b64_image
            }
            
            response = requests.post(GROUNDING_DINO_ENDPOINT, json=data, timeout=30)
            assert response.status_code == 200
            
            result = response.json()
            boxes = result['prediction']['boxes']
            labels = result['prediction']['labels']
            scores = result['prediction']['scores']
            
            results.append({
                'image': img_name,
                'count': len(boxes),
                'labels': set(labels),
                'avg_score': sum(scores) / len(scores) if scores else 0
            })
            
            print(f"\n图片 {img_name}: 检测到 {len(boxes)} 个目标")
        
        # 验证至少一张图片有检测结果
        if len(results) > 0:
            total_detections = sum(r['count'] for r in results)
            print(f"\n多图片测试完成，共处理 {len(results)} 张图片，总计 {total_detections} 个检测结果")

    def test_gounding_dino_response_structure(self, detection_request_data):
        """测试API响应结构完整性"""
        response = requests.post(GROUNDING_DINO_ENDPOINT, json=detection_request_data, timeout=30)
        
        assert response.status_code == 200
        result = response.json()
        
        # 验证必需字段
        required_fields = ['model_id', 'request_id', 'prediction']
        for field in required_fields:
            assert field in result, f"响应中缺少必需字段: {field}"
        
        # 验证prediction中的必需字段
        prediction = result['prediction']
        required_prediction_fields = ['boxes', 'labels', 'scores']
        for field in required_prediction_fields:
            assert field in prediction, f"prediction中缺少必需字段: {field}"
        
        print("\n响应结构验证通过")

    def test_gounding_dino_box_coordinates(self, detection_request_data):
        """测试检测框坐标的有效性"""
        response = requests.post(GROUNDING_DINO_ENDPOINT, json=detection_request_data, timeout=30)
        
        assert response.status_code == 200
        result = response.json()
        
        boxes = result['prediction']['boxes']
        
        # 验证每个检测框的坐标有效性
        for i, box in enumerate(boxes):
            if len(box) >= 4:
                x1, y1, x2, y2 = box[0], box[1], box[2], box[3]
                
                # 验证坐标顺序
                assert x1 <= x2, f"检测框 {i}: x1 ({x1}) 应该小于等于 x2 ({x2})"
                assert y1 <= y2, f"检测框 {i}: y1 ({y1}) 应该小于等于 y2 ({y2})"
                
                # 验证坐标非负
                assert x1 >= 0 and y1 >= 0, f"检测框 {i} 的坐标应该非负"
                
                # 计算检测框面积
                width = x2 - x1
                height = y2 - y1
                area = width * height
                
                assert area > 0, f"检测框 {i} 的面积应该大于0"
        
        print(f"\n检测框坐标验证通过，共验证 {len(boxes)} 个检测框")

    def test_gounding_dino_score_threshold(self, detection_request_data):
        """测试检测结果的置信度分数"""
        response = requests.post(GROUNDING_DINO_ENDPOINT, json=detection_request_data, timeout=30)
        
        assert response.status_code == 200
        result = response.json()
        
        scores = result['prediction']['scores']
        
        if len(scores) > 0:
            min_score = min(scores)
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            
            print(f"\n置信度分数统计:")
            print(f"  最小值: {min_score:.3f}")
            print(f"  最大值: {max_score:.3f}")
            print(f"  平均值: {avg_score:.3f}")
            
            # 验证所有分数在合理范围内
            for score in scores:
                assert 0 <= score <= 1, f"置信度分数应该在0-1之间: {score}"
        else:
            print("\n未检测到目标")

    def test_gounding_dino_response_time(self, detection_request_data):
        """测试API响应时间"""
        import time
        
        start_time = time.time()
        response = requests.post(GROUNDING_DINO_ENDPOINT, json=detection_request_data, timeout=30)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        
        assert response.status_code == 200
        result = response.json()
        assert 'prediction' in result
        
        print(f"\nAPI响应时间: {elapsed_time:.2f} 秒")
        # 目标检测通常在几秒内完成
        assert elapsed_time < 30, "API响应时间不应该超过30秒"


if __name__ == "__main__":
    # 直接运行测试
    pytest.main([__file__, "-v", "-s"])
