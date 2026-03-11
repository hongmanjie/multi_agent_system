import requests
import base64

# 模型服务地址
BASE_URL = "http://127.0.0.1:8010"

def image_to_base64(image_path):
    """
    将图片转为 Base64流
    :param image_path: 图片路径
    :return:
    """
    with open(image_path, "rb") as file:
        base64_data = base64.b64encode(file.read()).decode('utf-8')  # base64编码
    return base64_data


def test_face_detect_predict():
    """
    测试人脸检测模型
    """
    image_path = '../test_datas/retinaface_detection/baby_yangmi_nining.jpg'
    b64 = image_to_base64(image_path)
    data = {
        'request_id': "001",
        'image': b64,
        "nms_threshold": 0.5,
        "score_threshold": 0.9,
        "roi": []
    }

    # 测试 face_detection 模型
    model_id = "retinaface_detection"
    model_type = "cv"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"
    response = requests.post(url=url, json=data)
    print("Face Detection Response:")
    print(response.json())

    result = response.json()

    import cv2
    img_raw = cv2.imread(image_path)
    for b in result['prediction']['detections']:
        cv2.rectangle(img_raw, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)

    for b in result['prediction']['landmarks']:
        cv2.circle(img_raw, (b[0], b[1]), 1, (0, 0, 255), 4)
        cv2.circle(img_raw, (b[2], b[3]), 1, (0, 255, 255), 4)
        cv2.circle(img_raw, (b[4], b[5]), 1, (255, 0, 255), 4)
        cv2.circle(img_raw, (b[6], b[7]), 1, (0, 255, 0), 4)
        cv2.circle(img_raw, (b[8], b[9]), 1, (255, 0, 0), 4)

    # cv2.imshow("hahah", img_raw)
    # cv2.waitKey(0)
    cv2.imwrite("retinaface_result_from_api.jpg", img_raw)


def test_face_recognize_predict():
    """
    测试人脸识别模型
    """
    image_path = '../test_datas/arcface_recognition/Aaron_Peirsol/Aaron_Peirsol_0001.jpg'
    b64 = image_to_base64(image_path)
    data = {
        'request_id': "001",
        'image': b64
    }
    # 模型服务地址
    model_id = "arcface_recognition"
    model_type = "cv"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"

    response = requests.post(url=url, json=data)

    print("Face recgonize Response:")
    result = response.json()
    
    print(result["model_id"])
    emb = result["prediction"]["embedding"]
    print(f"Embedding: {emb}")
    print(f"Embedding shape: {len(emb)}")


def test_grounding_dino_predict():
    """
    测试grounding_dino模型
    """
    image_path = '../test_datas/yolo/bus.jpg'
    b64 = image_to_base64(image_path)
    data = {
        'request_id': "001",
        'image': b64
    }
    # 模型服务地址
    model_id = "grounding_dino"
    model_type = "cv"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"

    response = requests.post(url=url, json=data)

    print("GoundingDINO Response:")
    result = response.json()
    print(result)
    
    # 检查响应中是否包含错误
    if 'detail' in result:
        print(f"错误: {result['detail']}")
        return
    
    # 检查响应状态码
    if response.status_code != 200:
        print(f"请求失败，状态码: {response.status_code}")
        return
    
    # 检查是否包含 prediction 字段
    if 'prediction' not in result:
        print("响应中未找到 prediction 字段")
        print(f"完整响应: {result}")
        return
    
    boxes = result["prediction"]["boxes"]
    labels = result["prediction"]["labels"]
    scores = result["prediction"]["scores"]
    print("boxes:", boxes)
    print("labels:", labels)
    print("scores:", scores)


def test_minicpmv_4_5_generate():
    """
    minicpmv_4_5多模态模型测试
    """
    image_path = '../test_datas/retinaface_detection/baby_yangmi_nining.jpg'
    b64 = image_to_base64(image_path)
    data = {
        'request_id': "001",
        'image': b64,
        'text': "请描述一下这张图片的内容。",
        'temperature': 0.9
    }
    # 模型服务地址
    model_id = "minicpmv_4_5"
    model_type = "multi_modal"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"

    response = requests.post(url=url, json=data)

    print("minicpmv_4_5 Response:")
    result = response.json()
    print(result)



if __name__ == "__main__":
    # 根据model_server.json的配置进行测试，enable为true,调用对应model_id的测试函数
    ###########################
    model_id2test_func = {
        "retinaface_detection": test_face_detect_predict,
        "arcface_recognition": test_face_recognize_predict,
        "grounding_dino": test_grounding_dino_predict,
        "minicpmv_4_5": test_minicpmv_4_5_generate
    }


    config_path = "../configs/model_server.json"
    import json
    # 根据api服务启动的配置文件，选择需要测试的模型
    with open(config_path, "r", encoding="utf-8") as f:
        model_configs = json.load(f)

    for cfg in model_configs:
        model_id = cfg.get("model_id")

        if model_id in model_id2test_func:
            print(f"==========Testing model: {model_id}=============")
            test_func = model_id2test_func[model_id]
            test_func()
        else:
            # print(f"No test function defined for model: {model_id}")
            pass

