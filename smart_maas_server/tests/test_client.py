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


##### 测试 face_emotion 模型
def test_face_emotion_predict():
    pass


def test_facial_expression_recognition_predict():
    """
    测试面部表情识别模型
    """
    image_path = "../test_datas/face_emotion/angry.png"
    b64 = image_to_base64(image_path)
    data = {
        'request_id': "001",
        'image': b64
    }

    # 测试 facial_expression_recognition 模型
    model_id = "facial_expression_recognition"
    model_type = "cv"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"
    response = requests.post(url=url, json=data)
    print("Facial Expression Recognition Response:")
    result = response.json()
    print(result["prediction"]["score"])
    print(result["prediction"]["predicted"])

def test_yolov12_predict():
    """
    测试yolov12模型
    """
    image_path = '../test_datas/yolo/bus.jpg'
    b64 = image_to_base64(image_path)
    data = {
        'request_id': "001",
        'image': b64,
        "nms_threshold": 0.4,
        "score_threshold": 0.92,
        "roi":[
                [50.0, 60.0, 150.0, 200.0],  # 第一个区域
                [200.5, 180.3, 400.0, 350.9],  # 第二个区域
                [120.8, 300.2, 280.4, 420.5]   # 第三个区域
            ]
    }
    # 模型服务地址
    model_id = "yolov12"
    model_type = "cv"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"

    response = requests.post(url=url, json=data)

    print("Yolov12 Response:")
    result = response.json()
    print(result)
    boxes = result["prediction"]["boxes"]
    labels = result["prediction"]["labels"]
    scores = result["prediction"]["scores"]
    print("boxes:", boxes)
    print("labels:", labels)
    print("scores:", scores)

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


def extract_frame_from_video(video_path, frame_index=0, output_image_path=None):
    """
    从视频中提取一帧作为图片
    :param video_path: 视频文件路径
    :param frame_index: 要提取的帧索引（默认第一帧）
    :param output_image_path: 输出图片路径，如果为None则不保存
    :return: 提取的图片路径（如果保存）或None
    """
    try:
        import cv2
        import os
        
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


def test_minicpmv_4_5_video_summary():
    """
    minicpmv_4_5多模态模型测试 - 对视频进行总结摘要
    使用图片和提示词，让模型对视频 PW8Tl_BJNWA.mp4 进行总结摘要
    """
    # 视频文件名
    video_filename = "PW8Tl_BJNWA.mp4"
    video_path = video_filename  # 默认当前目录
    # 也可以使用完整路径，例如: '../test_datas/videos/PW8Tl_BJNWA.mp4'
    
    # 尝试从视频中提取关键帧作为图片
    image_path = None
    try:
        extracted_frame = extract_frame_from_video(video_path, frame_index=0)
        if extracted_frame:
            image_path = extracted_frame
            print(f"成功从视频提取关键帧: {image_path}")
    except Exception as e:
        print(f"从视频提取帧失败: {str(e)}")
    
    # 如果提取失败，使用默认图片
    if image_path is None:
        image_path = '../test_datas/retinaface_detection/baby_yangmi_nining.jpg'
        print(f"使用默认图片: {image_path}")
    
    # 将图片转为base64
    b64 = image_to_base64(image_path)
    
    # 构建提示词，说明要对视频进行总结摘要
    prompt_text = f"请对视频 {video_filename} 进行详细的总结和摘要。请描述视频的主要内容、关键场景、人物、动作和情节发展。"
    
    data = {
        'request_id': "video_summary_001",
        'image': b64,
        'text': prompt_text,
        'temperature': 0.9
    }
    
    # 模型服务地址
    model_id = "minicpmv_4_5"
    model_type = "multi_modal"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"

    print("=" * 60)
    print(f"正在请求视频摘要")
    print(f"视频文件名: {video_filename}")
    print(f"使用图片: {image_path}")
    print(f"提示词: {prompt_text}")
    print("=" * 60)
    
    response = requests.post(url=url, json=data, timeout=120)

    print("\n" + "=" * 60)
    print("minicpmv_4_5 视频摘要 Response:")
    print("=" * 60)
    result = response.json()
    print(result)
    
    # 提取并打印摘要结果
    if 'prediction' in result:
        summary = result['prediction']
        print("\n" + "=" * 60)
        print("视频摘要结果:")
        print("=" * 60)
        print(summary)
        print("=" * 60)
    else:
        print("响应中未找到预测结果")


def test_df5b_clip_generate():
    """
    df5b_clip多模态模型测试
    """
    image_path = '../test_datas/clip/1.png'
    b64 = image_to_base64(image_path)
    data = {
        'request_id': "001",
        'image': b64,
        'text': "蓝色小车"
    }
    # 模型服务地址
    model_id = "df5b_clip"
    model_type = "multi_modal"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"

    response = requests.post(url=url, json=data)

    print("df5b_clip Response:")
    result = response.json()
    print(len(result['prediction']['text_feat']))
    print(len(result['prediction']['image_feat']))

def test_qwen2_5_llm_generate():
    """
    qwen2_5_llm_transformer 服务测试
    """
    data = {
        'request_id': "001",
        'prompt': "请介绍几个多模态视觉大模型",
        'temperature': 0.7,
        'top_p': 0.9 ,
        'top_k': 20,
        'repetition_penalty': 1.05 ,
        'max_new_tokens': 512
    }
    # 模型服务地址
    model_id = "qwen2_5_llm"
    model_type = "nlp"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"

    response = requests.post(url=url, json=data)

    print("qwen2_5_llm Response:")
    result = response.json()
    print(result)

def test_qwen3_llm_generate():
    data = {
        'request_id': "001",
        'prompt': "请介绍几个多模态视觉大模型",
        'temperature': 0.7,
        'top_p': 0.9 ,
        'top_k': 20,
        'repetition_penalty': 1.05 ,
        'max_new_tokens': 512
    }
    # 模型服务地址
    model_id = "qwen3_llm"
    model_type = "nlp"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"

    response = requests.post(url=url, json=data)

    print("qwen3_llm Response:")
    result = response.json()
    print(result)

def test_qwen2_5_llm_vllm_generate():
    """
    qwen2_5_llm_vllm 服务测试
    """
    data = {
        'request_id': "001",
        'prompt': "请介绍几个多模态视觉大模型",
        'temperature': 0.7,
        'top_p': 0.9 ,
        'top_k': 20,
        'repetition_penalty': 1.05 ,
        'max_new_tokens': 512
    }
    # 模型服务地址
    model_id = "qwen2_5_llm_vllm"
    model_type = "nlp"
    url = f"{BASE_URL}/{model_type}/{model_id}/predict"

    response = requests.post(url=url, json=data)

    print("qwen2_5_llm_vllm Response:")
    result = response.json()
    print(result)


if __name__ == "__main__":
    # 根据model_server.json的配置进行测试，enable为true,调用对应model_id的测试函数
    ###########################
    model_id2test_func = {
        "retinaface_detection": test_face_detect_predict,
        # "face_emotion": test_face_emotion_predict,
        "arcface_recognition": test_face_recognize_predict,
        # "facial_expression_recognition": test_facial_expression_recognition_predict,
        # "yolov12": test_yolov12_predict,
        "grounding_dino": test_grounding_dino_predict,
        "minicpmv_4_5": test_minicpmv_4_5_generate
        # "df5b_clip": test_df5b_clip_generate,
        # "qwen2_5_llm": test_qwen2_5_llm_generate,
        # "qwen3_llm": test_qwen3_llm_generate,
        # "qwen2_5_llm_vllm": test_qwen2_5_llm_vllm_generate
    }

    # model_id2test_func = {
        # "yolov12": test_yolov12_predict,
        # "gounding_dino": test_gounding_dino_predict,
    #    "qwen2_5_vl": test_qwen2_5_vl_generate
    #    }

    config_path = "../configs/model_server.json"
    import json
    # 根据api服务启动的配置文件，选择需要测试的模型
    with open(config_path, "r", encoding="utf-8") as f:
        model_configs = json.load(f)

    for cfg in model_configs:
        model_id = cfg.get("model_id")
        #if not cfg.get("enabled", False):
        #    continue
        if model_id in model_id2test_func:
            print(f"==========Testing model: {model_id}=============")
            test_func = model_id2test_func[model_id]
            test_func()
        else:
            print(f"No test function defined for model: {model_id}")

