from re import S
import cv2
import json
import base64
import requests
import numpy as np
from typing import Dict, List, Any
import os
import time
from datetime import datetime
from difflib import SequenceMatcher
from pathlib import Path
import subprocess
import threading
import uuid
import asyncio
import aioboto3
import json
from llm import generate_text_by_qwen3_5



class VideoFrameAnalyzer:
    def __init__(self, base_url="http://127.0.0.1:8010", ffmpeg_path=None, frame_interval: int = 25, task_id = None, resource_id = None):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "VideoFrameAnalyzer/1.0.0"
        })
        self.frame_interval = frame_interval
        self.minface = 5000
        self.face_embeddings = []
        self.face_similarity_threshold = 0.4
        self.key_frames = []
        self.ffmpeg_path = ffmpeg_path or "ffmpeg"
        self.task_id = task_id
        self.resource_id = resource_id
        self.object_map = {
            "person": "人",
            "airplane": "飞机",
            "tank": "坦克",
            "helicopter": "直升机",
            "ship": "船舶",
            "armored vehicle": "装甲车",
            "missile": "导弹",
            "radar": "雷达"
        }
        self.keyword_code = {
            "氛围": "ATMOSPHERE",
            "情感感受": "EMOTION",
            "场景": "SCENE",
            "领域": "DOMAIN",
            "时代": "ERA",
            "拍摄角度": "SHOOTING_ANGLE",
            "主题": "THEME",
            "风格": "STYLE",
            "视角": "VIEWING_ANGLE",
            "事件行为": "EVENT",
            "时间段": "TIME_SLOT"
        }

    
    def image_to_base64(self, image_array):
        if image_array is None:
            return None
        try:
            success, encoded_image = cv2.imencode('.jpg', image_array)
            if success:
                base64_data = base64.b64encode(encoded_image).decode('utf-8')
                return base64_data
            return None
        except Exception as e:
            print(f"图像编码错误: {e}")
            return None
    
    def calculate_frame_similarity(self, frame1, frame2):
        """
        计算两帧之间的相似度
        
        Args:
            frame1: 第一帧图像
            frame2: 第二帧图像
            
        Returns:
            相似度值，0-1之间，值越大表示越相似
        """
        try:
            # 转换为灰度图
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # 调整大小以提高计算速度
            size = (256, 256)
            gray1 = cv2.resize(gray1, size)
            gray2 = cv2.resize(gray2, size)
            
            # 使用结构相似性指数(SSIM)计算相似度
            from skimage.metrics import structural_similarity as ssim
            similarity = ssim(gray1, gray2)
            
            return similarity
        except Exception as e:
            print(f"计算帧相似度错误: {e}")
            return 0.0
    
    async def upload_image_to_rustfs(self, image_data, object_name: str, bucket_name: str = "video-lake", is_path: bool = True) -> str:
        """
        将图片上传到RustFS对象存储
        
        Args:
            image_data: 本地图片路径或内存中的图像数据
            object_name: 对象名称
            bucket_name: 存储桶名称，默认为"video-lake"
            is_path: 是否为本地文件路径
            
        Returns:
            上传后的图片URL
        """
        import time
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                if is_path:
                    print(f"开始上传图片到RustFS: {image_data}")
                else:
                    print(f"开始上传内存中的图片到RustFS: {object_name}")
                
                # 初始化aioboto3会话
                session = aioboto3.Session()
                
                # 连接到RustFS
                async with session.client(
                    's3',
                    endpoint_url='http://192.168.86.113:9900',
                    aws_access_key_id='rustfsadmin',
                    aws_secret_access_key='rustfsadmin'
                ) as client:
                    # 检查存储桶是否存在，不存在则创建
                    try:
                        await client.head_bucket(Bucket=bucket_name)
                        print(f"存储桶 {bucket_name} 已存在")
                    except:
                        print(f"存储桶 {bucket_name} 不存在，正在创建...")
                        await client.create_bucket(Bucket=bucket_name)
                        print(f"存储桶 {bucket_name} 创建成功")
                    
                    # 上传图片
                    if is_path:
                        # 从本地文件上传
                        with open(image_data, 'rb') as f:
                            await client.upload_fileobj(f, bucket_name, object_name)
                    else:
                        # 从内存中的图像数据上传
                        # 将图像编码为JPEG格式
                        success, encoded_image = cv2.imencode('.jpg', image_data)
                        if success:
                            # 创建一个内存文件对象
                            import io
                            image_stream = io.BytesIO(encoded_image)
                            # 上传内存中的图像
                            await client.upload_fileobj(image_stream, bucket_name, object_name)
                        else:
                            raise Exception("无法编码图像数据")
                    
                    # 生成上传后的URL
                    image_url = f"http://192.168.86.113:9900/{bucket_name}/{object_name}"
                    print(f"图片上传成功，URL: {image_url}")
                    
                    return image_url
                    
            except Exception as e:
                print(f"上传图片到RustFS失败 (尝试 {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    print(f"等待 {retry_delay} 秒后重试...")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= 2  # 指数退避
                else:
                    # 所有重试都失败了，返回错误信息
                    print(f"经过 {max_retries} 次尝试后仍然上传失败")
                    # 如果上传失败，返回错误信息
                    return f"error://{str(e)}"
    
    async def push_key_frames_to_api(self, task_id: int, resource_id: int) -> bool:
        """
        将关键帧数据推送到指定的API端点
        
        Args:
            task_id: 任务ID
            video_name: 视频名称
            
        Returns:
            是否推送成功
        """
        try:
            print(f"开始推送关键帧数据，任务ID: {task_id}")
            
            # 准备关键帧数据
            frames = []
            for i, key_frame in enumerate(self.key_frames):
                # 上传关键帧图像到RustFS
                object_name = f"catalog_data_v3/resource_{resource_id}/key_frames/keyframe_{i}.jpg"
                key_frame_url = await self.upload_image_to_rustfs(key_frame['frame_data'], object_name, is_path=False)
                
                # 使用clip模型获取关键帧向量
                vector = []
                try:
                    # 将帧数据转换为base64编码
                    frame_b64 = self.image_to_base64(key_frame['frame_data'])
                    if frame_b64:
                        # 调用clip模型获取向量
                        clip_result = self.clip_vector(frame_b64)
                        if clip_result:
                            vector = clip_result.get('image_feat', [])
                            print(f"  成功获取关键帧向量，维度: {len(vector)}")
                        else:
                            print("  获取clip模型结果失败")
                    else:
                        print("  帧数据编码失败")
                except Exception as e:
                    print(f"  获取关键帧向量时出错: {e}")
                
                # 构建关键帧数据
                frame_data = {
                    "keyFrameUrl": key_frame_url,
                    "timestamp": key_frame.get('timestamp_seconds', 0.0),
                    "vector": vector
                }
                frames.append(frame_data)
            
            # 构建请求体
            payload = {
                "taskId": task_id,
                "frames": frames
            }
            
            print(f"准备推送 {len(frames)} 个关键帧数据")
            
            # 推送数据到API端点
            import requests
            api_url = "http://192.168.86.113:8081/api/v1/analysis/callback/key-frames"
            response = requests.post(api_url, json=payload, timeout=30)
            
            # 检查响应
            if response.status_code == 200:
                print(f"关键帧数据推送成功")
                print(f"API响应: {response.json()}")
                return True
            else:
                print(f"关键帧数据推送失败，状态码: {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
        except Exception as e:
            print(f"推送关键帧数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def crop_image(self, image, box):
        """
        根据边界框裁剪图像
        
        Args:
            image: 原始图像
            box: 边界框 [x1, y1, x2, y2]
            
        Returns:
            裁剪后的图像
        """
        if image is None or not box:
            return None
        
        try:
            x1, y1, x2, y2 = map(int, box)
            # 确保坐标在图像范围内
            h, w = image.shape[:2]
            x1 = max(0, min(x1, w))
            y1 = max(0, min(y1, h))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))
            
            if x2 > x1 and y2 > y1:
                return image[y1:y2, x1:x2]
            return None
        except Exception as e:
            print(f"裁剪图像失败: {e}")
            return None
    
    def extract_key_frames(self, video_path: str, frame_interval: int = 25) -> List[Dict[str, Any]]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"无法打开视频文件: {video_path}")
        
        all_frames = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        duration = total_frames / fps if fps > 0 else 0
        
        print(f"视频信息:")
        print(f"  总帧数: {total_frames}")
        print(f"  帧率: {fps:.2f} fps")
        print(f"  时长: {duration:.2f} 秒")
        print(f"  抽帧间隔: 每 {frame_interval} 帧取一帧")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % frame_interval == 0:
                timestamp_seconds = frame_count / fps if fps > 0 else 0
                timestamp_formatted = f"{int(timestamp_seconds // 60)}:{int(timestamp_seconds % 60):02d}"
                
                all_frames.append({
                    "frame_index": frame_count,
                    "frame_data": frame,
                    "timestamp_seconds": round(timestamp_seconds, 2),
                    "timestamp_formatted": timestamp_formatted,
                    "face_detections": [],
                    "object_detections": [],
                    "scene_description": ""
                })
            
            frame_count += 1
        
        cap.release()
        
        print(f"成功提取 {len(all_frames)} 个视频帧")
        return all_frames
    
    def detect_faces(self, frame_b64: str) -> Dict[str, Any]:
        data = {
            'request_id': f"face_detect_{int(time.time())}",
            'image': frame_b64,
            "nms_threshold": 0.8,
            "score_threshold": 0.8
        }
        
        url = f"{self.base_url}/cv/retinaface_detection/predict"
        try:
            response = self.session.post(url=url, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and 'prediction' in result:
                    return result['prediction']
            return None
        except Exception as e:
            print(f"人脸检测错误: {e}")
            return None
    
    def recognize_faces(self, frame_b64: str) -> Dict[str, Any]:
        data = {
            'request_id': f"face_recognize_{int(time.time())}",
            'image': frame_b64
        }
        
        url = f"{self.base_url}/cv/arcface_recognition/predict"
        try:
            response = self.session.post(url=url, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and 'prediction' in result:
                    return result['prediction']
            return None
        except Exception as e:
            print(f"人脸检测错误: {e}")
            return None

    def clip_vector(self, frame_b64: str) -> Dict[str, Any]:
        data = {
            'request_id': f"dfn5b_clip{int(time.time())}",
            'image': frame_b64
        }
        
        url = f"{self.base_url}/multi_modal/dfn5b_clip/predict"
        try:
            response = self.session.post(url=url, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and 'prediction' in result:
                    return result['prediction']
            return None
        except Exception as e:
            print(f"人脸检测错误: {e}")
            return None

    def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        if not embedding1 or not embedding2 or len(embedding1) != len(embedding2):
            return 0.0
        
        vec1 = np.array(embedding1)
        vec2 = np.array(embedding2)
        
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = np.dot(vec1, vec2) / (norm1 * norm2)
        return float(similarity)
    
    def match_face_embedding(self, embedding: List[float]) -> int:
        if not self.face_embeddings:
            face_info = {
                "name": "未知人脸0",
                "embedding": embedding
            }
            self.face_embeddings.append(face_info)
            return self.face_embeddings[0].get("name")
        
        max_similarity = 0.0
        best_match_index = -1
        
        for idx, existing_embedding_item in enumerate(self.face_embeddings):
            similarity = self.compute_similarity(embedding, existing_embedding_item.get("embedding"))
            if similarity > max_similarity:
                max_similarity = similarity
                best_match_index = idx
        
        if max_similarity >= self.face_similarity_threshold:
            return self.face_embeddings[best_match_index].get("name")
        else:
            face_info = {
                "name": f"未知人脸{len(self.face_embeddings)}",
                "embedding": embedding
            }
            self.face_embeddings.append(face_info)
            return self.face_embeddings[-1].get("name")
    
    def detect_objects(self, frame_b64: str) -> Dict[str, Any]:
        data = {
            'request_id': f"object_detect_{int(time.time())}",
            'image': frame_b64
            # "categories": ["person", "airplane", "tank", "helicopter", "ship", "armored vehicle", "missile", "radar"],
            # "score_threshold": 0.8,
            # "nms_threshold": 0.8,
        }
        
        url = f"{self.base_url}/cv/grounding_dino_v1/predict"
        try:
            response = self.session.post(url=url, json=data, timeout=30)
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, dict) and 'prediction' in result:
                    return result['prediction']
            return None
        except Exception as e:
            print(f"目标检测错误: {e}")
            return None
    
    def analyze_scene(self, frame_b64: str, prompt: str) -> Dict[str, Any]:
        data = {
            'request_id': f"scene_analyze_{int(time.time())}",
            'image': frame_b64,
            'text': prompt,
            'temperature': 0.3
        }
        
        url = f"{self.base_url}/multi_modal/minicpmv_4_5/predict"
        try:
            response = self.session.post(url=url, json=data, timeout=60)
            if response.status_code == 200:
                vl_result = response.json()
                if isinstance(vl_result, dict) and 'prediction' in vl_result:
                    return vl_result['prediction']
        except Exception as e:
            print(f"多模态分析错误: {e}")
            return None
    
    def compute_visual_similarity(self, frame1: np.ndarray, frame2: np.ndarray) -> float:
        if frame1 is None or frame2 is None:
            return 0.0
        
        try:
            frame1_gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            frame2_gray = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            frame1_resized = cv2.resize(frame1_gray, (64, 64))
            frame2_resized = cv2.resize(frame2_gray, (64, 64))
            
            hist1 = cv2.calcHist([frame1_resized], [0], None, [16], [0, 256])
            hist2 = cv2.calcHist([frame2_resized], [0], None, [16], [0, 256])
            
            hist1 = hist1.flatten()
            hist2 = hist2.flatten()
            
            hist1 = hist1 / (hist1.sum() + 1e-10)
            hist2 = hist2 / (hist2.sum() + 1e-10)
            
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return max(0.0, float(correlation))
        except Exception as e:
            print(f"视觉相似度计算错误: {e}")
            return 0.0
    
    def compute_object_similarity(self, objects1: List[Dict], objects2: List[Dict]) -> float:
        if not objects1 or not objects2:
            return 0.0
        
        names1 = set(obj.get('name', '') for obj in objects1)
        names2 = set(obj.get('name', '') for obj in objects2)
        
        if not names1 or not names2:
            return 0.0
        
        intersection = len(names1 & names2)
        union = len(names1 | names2)
        
        return intersection / union if union > 0 else 0.0
    
    def compute_weighted_scene_similarity(self, frame1: Dict, frame2: Dict) -> float:
        weights = {
            'visual': 0.5,
            'objects': 0.3,
            'faces': 0.2
        }
        
        visual_sim = self.compute_visual_similarity(
            frame1.get('frame_data'),
            frame2.get('frame_data')
        )
        
        object_sim = self.compute_object_similarity(
            frame1.get('object_detections', ''),
            frame2.get('object_detections', '')
        )
        
        faces1 = frame1.get('face_detections', [])
        faces2 = frame2.get('face_detections', [])
        names1 = set(f.get('name', '') for f in faces1)
        names2 = set(f.get('name', '') for f in faces2)
        
        if not names1 or not names2:
            face_sim = 0.0
        else:
            face_sim = len(names1 & names2) / len(names1 | names2)
        
        weighted_similarity = (
            weights['visual'] * visual_sim +
            weights['objects'] * object_sim +
            weights['faces'] * face_sim
        )
        
        return weighted_similarity
    
    def segment_scenes(self, analyzed_frames: List[Dict[str, Any]], 
                       similarity_threshold: float = 0.65,  # 提高阈值减少场景数量
                       min_scene_frames: int = 20) -> List[Dict[str, Any]]:  # 最小场景帧数
        if not analyzed_frames:
            return []
        
        print(f"\n{'='*60}")
        print(f"开始场景划分，共 {len(analyzed_frames)} 帧")
        print(f"相似度阈值: {similarity_threshold}")
        print(f"最小场景帧数: {min_scene_frames}")
        print(f"{'='*60}\n")
        
        scenes = []
        current_scene = [analyzed_frames[0]]
        scene_id = 0
        
        for i in range(1, len(analyzed_frames)):
            prev_frame = analyzed_frames[i-1]
            curr_frame = analyzed_frames[i]
            
            similarity = self.compute_weighted_scene_similarity(prev_frame, curr_frame)
            
            print(f"[{i}/{len(analyzed_frames)}] 帧间相似度: {similarity:.3f}")
            
            # 场景分割条件：
            # 1. 场景帧数足够多
            # 2. 当前帧与前两帧的相似度低
            # 3. 当前帧与后两帧的相似度高
            should_split = False
            
            if len(current_scene) >= min_scene_frames:
                # 计算当前帧与前两帧的相似度
                prev_sim1 = similarity  # 当前帧与前一帧的相似度
                prev_sim2 = 0.0
                if i >= 2:
                    prev_sim2 = self.compute_weighted_scene_similarity(analyzed_frames[i-2], curr_frame)
                
                # 计算当前帧与后两帧的相似度
                next_sim1 = 0.0
                next_sim2 = 0.0
                if i + 1 < len(analyzed_frames):
                    next_sim1 = self.compute_weighted_scene_similarity(curr_frame, analyzed_frames[i+1])
                if i + 2 < len(analyzed_frames):
                    next_sim2 = self.compute_weighted_scene_similarity(curr_frame, analyzed_frames[i+2])
                
                # 打印相似度信息
                print(f"  前两帧相似度: {prev_sim1:.3f}, {prev_sim2:.3f}")
                print(f"  后两帧相似度: {next_sim1:.3f}, {next_sim2:.3f}")
                
                # 分割条件：
                # 1. 当前帧与前两帧的相似度都低于阈值
                # 2. 当前帧与后两帧的相似度都高于阈值
                # 3. 场景帧数足够多
                prev_similarity_low = (prev_sim1 < similarity_threshold and (i < 2 or prev_sim2 < similarity_threshold))
                next_similarity_high = (i + 1 >= len(analyzed_frames) or next_sim1 >= similarity_threshold) and (i + 2 >= len(analyzed_frames) or next_sim2 >= similarity_threshold)
                
                if prev_similarity_low and next_similarity_high:
                    should_split = True
                    print(f"  满足分割条件，执行场景分割")
            
            if should_split:
                scene_start_frame = current_scene[0]['frame_index']
                scene_end_frame = current_scene[-1]['frame_index']
                scene_start_time = current_scene[0]['timestamp_seconds']
                scene_end_time = current_scene[-1]['timestamp_seconds']
                
                scene_info = {  
                    'scene_id': scene_id,
                    'start_frame': scene_start_frame,
                    'end_frame': scene_end_frame,
                    'start_time': scene_start_time,
                    'end_time': scene_end_time,
                    'duration': scene_end_time - scene_start_time,
                    'frame_list': [f['frame_data'] for f in current_scene],
                    'scene_descriptions': [f.get('scene_description', '') for f in current_scene if f.get('scene_description')],
                    'objects': list(set(obj.get('name', '') for f in current_scene for obj in f.get('object_detections', []))),
                    'faces': list(set(face.get('name', '') for f in current_scene for face in f.get('face_detections', [])))
                }
                
                scenes.append(scene_info)
                print(f"  场景 {scene_id}: 帧 {scene_start_frame}-{scene_end_frame}, 时长 {scene_end_time - scene_start_time:.2f}s, {len(current_scene)} 帧")
                
                scene_id += 1
                current_scene = [curr_frame]
            else:
                current_scene.append(curr_frame)
        
        if current_scene:
            scene_start_frame = current_scene[0]['frame_index']
            scene_end_frame = current_scene[-1]['frame_index']
            scene_start_time = current_scene[0]['timestamp_seconds']
            scene_end_time = current_scene[-1]['timestamp_seconds']
            
            scene_info = {
                'scene_id': scene_id,
                'start_frame': scene_start_frame,
                'end_frame': scene_end_frame,
                'start_time': scene_start_time,
                'end_time': scene_end_time,
                'duration': scene_end_time - scene_start_time,
                'frame_list': [f['frame_data'] for f in current_scene],
                'scene_descriptions': [f.get('scene_description', '') for f in current_scene if f.get('scene_description')],
                'objects': list(set(obj.get('name', '') for f in current_scene for obj in f.get('object_detections', []))),
                'faces': list(set(face.get('name', '') for f in current_scene for face in f.get('face_detections', [])))
            }
            
            scenes.append(scene_info)
            print(f"  场景 {scene_id}: 帧 {scene_start_frame}-{scene_end_frame}, 时长 {scene_end_time - scene_start_time:.2f}s, {len(current_scene)} 帧")
        
        # 合并相似场景
        scenes = self.merge_similar_scenes(scenes, similarity_threshold=0.7)
        
        print(f"\n{'='*60}")
        print(f"场景划分完成，共 {len(scenes)} 个场景")
        print(f"{'='*60}\n")
        
        return scenes
    
    def merge_similar_scenes(self, scenes: List[Dict[str, Any]], similarity_threshold: float) -> List[Dict[str, Any]]:
        """
        合并相似的相邻场景
        """
        if len(scenes) < 2:
            return scenes
        
        merged_scenes = []
        i = 0
        
        while i < len(scenes):
            current_scene = scenes[i]
            
            # 检查下一个场景是否相似
            if i + 1 < len(scenes):
                next_scene = scenes[i + 1]
                scene_similarity = self.compute_scene_similarity(current_scene, next_scene)
                
                print(f"场景 {current_scene['scene_id']} 和 {next_scene['scene_id']} 相似度: {scene_similarity:.3f}")
                
                # 如果相似度高于阈值，合并场景
                if scene_similarity >= similarity_threshold:
                    merged_scene = {
                        'scene_id': current_scene['scene_id'],
                        'start_frame': current_scene['start_frame'],
                        'end_frame': next_scene['end_frame'],
                        'start_time': current_scene['start_time'],
                        'end_time': next_scene['end_time'],
                        'duration': next_scene['end_time'] - current_scene['start_time'],
                        'frame_list': current_scene['frame_list'] + next_scene['frame_list'],
                        'scene_descriptions': current_scene['scene_descriptions'] + next_scene['scene_descriptions'],
                        'objects': list(set(current_scene['objects'] + next_scene['objects'])),
                        'faces': list(set(current_scene['faces'] + next_scene['faces']))
                    }
                    
                    merged_scenes.append(merged_scene)
                    print(f"  合并场景 {current_scene['scene_id']} 和 {next_scene['scene_id']}")
                    i += 2  # 跳过已合并的下一个场景
                    continue
            
            # 如果没有合并，直接添加当前场景
            merged_scenes.append(current_scene)
            i += 1
        
        # 更新场景ID
        for idx, scene in enumerate(merged_scenes):
            scene['scene_id'] = idx
        
        # 检查最后一个场景的时长，若不足10秒则与前一个场景合并
        if len(merged_scenes) >= 2:
            last_scene = merged_scenes[-1]
            last_scene_duration = last_scene.get('duration', 0)
            
            if last_scene_duration < 10.0:
                print(f"最后一个场景时长 {last_scene_duration:.2f} 秒，不足10秒，与前一个场景合并")
                
                # 获取倒数第二个场景
                prev_scene = merged_scenes[-2]
                
                # 合并场景
                merged_scene = {
                    'scene_id': prev_scene['scene_id'],
                    'start_frame': prev_scene['start_frame'],
                    'end_frame': last_scene['end_frame'],
                    'start_time': prev_scene['start_time'],
                    'end_time': last_scene['end_time'],
                    'duration': last_scene['end_time'] - prev_scene['start_time'],
                    'frame_list': prev_scene['frame_list'] + last_scene['frame_list'],
                    'scene_descriptions': prev_scene['scene_descriptions'] + last_scene['scene_descriptions'],
                    'objects': list(set(prev_scene['objects'] + last_scene['objects'])),
                    'faces': list(set(prev_scene['faces'] + last_scene['faces']))
                }
                
                # 替换最后两个场景为合并后的场景
                merged_scenes[-2] = merged_scene
                merged_scenes.pop()
                
                # 更新后续场景的ID
                for i in range(len(merged_scenes)):
                    merged_scenes[i]['scene_id'] = i
                
                print(f"场景合并完成，当前场景数: {len(merged_scenes)}")
        
        return merged_scenes
    
    def compute_scene_similarity(self, scene1: Dict[str, Any], scene2: Dict[str, Any]) -> float:
        """
        计算两个场景之间的相似度
        """
        # 计算物体相似度
        objects1 = set(scene1.get('objects', []))
        objects2 = set(scene2.get('objects', []))
        
        if objects1 or objects2:
            object_sim = len(objects1 & objects2) / len(objects1 | objects2)
        else:
            object_sim = 0.0
        
        # 计算人脸相似度
        faces1 = set(scene1.get('faces', []))
        faces2 = set(scene2.get('faces', []))
        
        if faces1 or faces2:
            face_sim = len(faces1 & faces2) / len(faces1 | faces2)
        else:
            face_sim = 0.0
        
        # 加权计算总相似度
        weighted_similarity = 0.6 * object_sim + 0.4 * face_sim
        
        return weighted_similarity
    
    def extract_scene_audio(self, video_path: str, start_time: float, end_time: float, output_path: str) -> bool:
        """
        从视频中提取指定时间段的音频
        
        Args:
            video_path: 视频文件路径
            start_time: 开始时间（秒）
            end_time: 结束时间（秒）
            output_path: 输出音频文件路径
            
        Returns:
            是否成功提取
        """
        try:
            # 确保输出目录存在
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # 检查 ffmpeg 是否可用
            try:
                result = subprocess.run(
                    [self.ffmpeg_path, '-version'],
                    capture_output=True,
                    text=True
                )
                if result.returncode != 0:
                    print(f"    ✗ ffmpeg 不可用，请确保已安装并添加到环境变量")
                    return False
            except FileNotFoundError:
                print(f"    ✗ 找不到 ffmpeg 命令: {self.ffmpeg_path}")
                print(f"    请下载 ffmpeg 并添加到环境变量，或在初始化时指定 ffmpeg_path 参数")
                return False
            
            # 使用 ffmpeg 提取音频
            cmd = [
                self.ffmpeg_path, '-y',
                '-i', video_path,
                '-ss', f'{start_time}',
                '-to', f'{end_time}',
                '-vn',  # 只提取音频
                '-acodec', 'libmp3lame',  # MP3 编码
                '-ab', '128k',  # 音频比特率
                output_path
            ]
            
            print(f"  提取音频: {output_path}")
            print(f"  执行命令: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"    ✓ 音频提取成功")
                return True
            else:
                print(f"    ✗ 音频提取失败: {result.stderr}")
                return False
        except Exception as e:
            print(f"    ✗ 音频提取异常: {e}")
            return False
    
    def transcribe_scene_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        转录场景音频
        
        Args:
            audio_path: 音频文件路径
            
        Returns:
            识别结果
        """
        if not os.path.exists(audio_path):
            print(f"    ✗ 音频文件不存在: {audio_path}")
            return {"success": False, "error": "音频文件不存在"}
        
        try:
            # 读取音频文件并编码为base64
            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            
            # 编码为base64
            import base64
            audio_data_base64 = base64.b64encode(audio_data).decode('utf-8')
            
            # 获取音频格式
            audio_format = Path(audio_path).suffix.lstrip('.').lower()
            if audio_format not in ['wav', 'mp3']:
                audio_format = 'wav'
            
            # 构建请求数据
            import uuid
            request_id = str(uuid.uuid4())
            data = {
                'request_id': request_id,
                'audio_data': audio_data_base64,
                'audio_format': audio_format
            }
            
            # 发送请求到smart_maas_server的funasr接口
            response = self.session.post(
                "http://127.0.0.1:8010/audio/funasr_asr/predict",
                json=data,
                headers={"Content-Type": "application/json"},
                timeout=60  # 60秒超时
            )
            
            if response.status_code == 200:
                response_data = response.json()
                
                # 从响应中提取预测结果
                if 'prediction' in response_data:
                    result = response_data['prediction']
                else:
                    result = response_data
                
                print(f"    ✓ 语音识别完成")
                
                return result
            else:
                error_msg = f"识别失败: {response.status_code} - {response.text}"
                print(f"    ✗ {error_msg}")
                return {"success": False, "error": error_msg}
                
        except requests.exceptions.Timeout:
            error_msg = "请求超时，请检查音频文件大小或网络连接"
            print(f"    ✗ {error_msg}")
            return {"success": False, "error": error_msg}
        except Exception as e:
            error_msg = f"请求异常: {str(e)}"
            print(f"    ✗ {error_msg}")
            return {"success": False, "error": error_msg}
    
    def process_scenes_audio(self, video_path: str, scenes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        处理所有场景的音频（修正：提前初始化audio_results，避免未定义报错）
        """
        print(f"\n{'='*60}")
        print(f"开始处理场景音频，共 {len(scenes)} 个场景")
        print(f"{'='*60}\n")
        
        # 提前初始化audio_results，避免未定义报错（核心修改1）
        audio_results = []
        
        # 创建音频输出目录
        video_dir = os.path.dirname(video_path)
        video_name = Path(video_path).stem
        audio_output_dir = os.path.join(video_dir, f"{video_name}_scenes_audio")
        os.makedirs(audio_output_dir, exist_ok=True)
        
        # 首先提取整条视频的完整音频
        full_audio_filename = f"{video_name}_full.mp3"
        full_audio_path = os.path.join(audio_output_dir, full_audio_filename)
        
        print(f"提取完整视频音频: {full_audio_path}")
        
        # 提取完整音频（从0秒到视频结束）
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_duration = total_frames / fps if fps > 0 else 0
        cap.release()
        
        extract_success = self.extract_scene_audio(video_path, 0, total_duration, full_audio_path)
        
        if extract_success:
            # 对完整音频进行语音识别
            print(f"对完整音频进行语音识别...")
            full_transcription_result = self.transcribe_scene_audio(full_audio_path)
            
            if full_transcription_result.get('success', True):
                print(f"完整音频识别成功")
                # 修正：audio_results 赋值为列表，确保后续遍历正常（核心修改2）
                audio_results = full_transcription_result.get('results', [])
                
                # 根据场景拆分识别结果
                for i, scene in enumerate(scenes):
                    scene_id = scene['scene_id']
                    start_time = scene['start_time']
                    end_time = scene['end_time']
                    
                    print(f"[{i+1}/{len(scenes)}] 处理场景 {scene_id}")
                    print(f"  时间范围: {start_time:.2f}s - {end_time:.2f}s")
                    
                    scene_text = ''
                    if len(audio_results) > 0:
                        for result in audio_results:
                            # 修正：增加键存在判断，避免KeyError（核心修改3）
                            result_start = result.get('start_time', 0.0)
                            result_end = result.get('end_time', 0.0)
                            speaker_id = result.get('speaker_id', 'unknown')
                            result_text = result.get('text', '')
                            
                            # 检查该结果是否在当前场景的时间范围内
                            if (result_start >= start_time and result_start <= end_time) or \
                            (result_end >= start_time and result_end <= end_time) or \
                            (result_start <= start_time and result_end >= end_time):
                                scene_text += f" 发言人{speaker_id}: {result_text}"
                    else:
                        # 如果没有时间戳信息，使用完整文本
                        scene['audio_path'] = ""
                        scene['transcription_text'] = ""
                        scene['transcription_path'] = ""  # 简单处理，实际需要根据时间拆分
                    
                    # 更新场景信息
                    scene['audio_path'] = full_audio_path
                    scene['transcription_text'] = scene_text
                    scene['transcription_path'] = os.path.join(audio_output_dir, f"{full_audio_filename.split('.')[0]}_语音识别.json")
                    
                    if scene_text:
                        print(f"    ✓ 识别文本: {scene_text[:100]}...")
                    else:
                        print(f"    ✓ 识别完成，无文本内容")
                    
                    print()
            else:
                print(f"    ✗ 完整音频语音识别失败")
                # 直接将所有场景的音频相关字段赋值为空
                for scene in scenes:
                    scene['audio_path'] = ""
                    scene['transcription_text'] = ""
                    scene['transcription_path'] = ""
                print("  已将所有场景的音频相关字段设置为空")
        else:
            print(f"    ✗ 提取完整音频失败")
            # 直接将所有场景的音频相关字段设置为空
            for scene in scenes:
                scene['audio_path'] = ""
                scene['transcription_text'] = ""
                scene['transcription_path'] = ""
            print("  已将所有场景的音频相关字段设置为空")
        
        print(f"{'='*60}")
        print(f"场景音频处理完成")
        print(f"{'='*60}\n")
        
        # 清理音频文件和目录
        try:
            # 删除提取的音频文件
            if 'full_audio_path' in locals() and os.path.exists(full_audio_path):
                os.remove(full_audio_path)
                print(f"  已删除音频文件: {full_audio_path}")
            
            # 删除音频输出目录
            if 'audio_output_dir' in locals() and os.path.exists(audio_output_dir):
                os.rmdir(audio_output_dir)
                print(f"  已删除音频输出目录: {audio_output_dir}")
        except Exception as e:
            print(f"  清理音频文件时发生错误: {e}")
        
        # 返回值：确保audio_results始终是列表，避免类型错误
        return audio_results, scenes
    
    def generate_scene_summary(self, scene: Dict[str, Any]) -> Dict[str, Any]:
        """
        生成场景摘要
        
        Args:
            scene: 场景信息
            
        Returns:
            包含摘要的场景信息
        """
        # 获取场景描述和语音内容
        # scene_descriptions = scene.get('scene_descriptions', [])
        speech_content = scene.get('transcription_text', '')
        frame_list = scene.get('frame_list', [])

        
        # 构建提示词
        prompt_template = """你是一位专业的视频内容分析专家。请根据提供的视频片段信息，包括语音转文字内容和片段帧数据，这些内容出自一个视频场景，完成以下任务：

## 输入信息说明
- 片段帧数据：已通过API参数`frame_data`直接传递，请基于这些图像帧进行视觉内容分析
- 语音转文字内容：如下所示，与帧数据属于同一时间段

## 语音转文字内容：
{speech_content}

## 输出要求
请严格按照以下固定格式输出，**必须包含两个部分，顺序不能颠倒**：

**场景摘要：**
[仅基于片段帧的视觉内容进行总结，描述画面中的人物、场景、动作等视觉信息，按时间顺序组织，如内容重复则进行概括。要求客观描述视觉内容，字数不宜过多，不要加入语音内容。]

**片段智读：**
[结合片段帧的视觉内容和语音转文字内容进行综合分析，以语音信息为主导，同时参考视觉信息进行补充，按时间顺序组织内容。]

注意：请务必按照上述格式输出，不要添加任何其他解释或前缀。"""
        # 格式化提示词
        formatted_prompt = prompt_template.format(
            speech_content=speech_content if speech_content else "无"
        )
        
        # 调用LLM模型
        try:
            # 服务地址
            BASE_URL = "http://192.168.86.113:8000"
            url = f"{BASE_URL}/analyze-video-by-frames"
            
            # 从frame_list中每隔5帧抽取一帧构成frame_data列表
            sampled_frames = []
            for i in range(0, len(frame_list), 10):
                frame_item = frame_list[i]
                # 将numpy数组转换为列表格式以供JSON序列化
                frame_data = frame_item.tolist()  # numpy array转换为嵌套列表
                sampled_frames.append(frame_data)
            print(f"  采样帧数据数量: {len(sampled_frames)}")
            # 构造请求数据
            data = {
                'frame_data': sampled_frames,  # 隔5帧抽取的帧数据列表
                'prompt': formatted_prompt
            }
            
            print(f"  生成场景摘要...")
            response = self.session.post(url, json=data, timeout=120)
            
            if response.status_code == 200:
                result = response.json()
                
                if 'result' in result:
                    summary = result['result']
                    print(f"    ✓ 场景摘要生成成功: {summary}...")

                    # 拆分摘要为场景摘要和片段智读
                    import re
                    # 提取场景摘要部分
                    scene_summary_match = re.search(r'\*\*场景摘要：\*\*(.*?)\*\*片段智读：\*\*', summary, re.DOTALL)
                    if scene_summary_match:
                        scene_summary_text = scene_summary_match.group(1).strip()
                        # 提取片段智读部分
                        segment_insight_text = summary[scene_summary_match.end():].strip()
                        # 存储到对应的字段
                        scene['scene_summary'] = scene_summary_text
                        scene['smart_read'] = segment_insight_text
                    else:
                        # 如果正则匹配失败，尝试其他方式提取
                        print(f"    ⚠ 正则匹配失败，尝试备用方案")
                        scene['scene_summary'] = ""
                        scene['smart_read'] = ""
                    print(f"    ✓ 摘要生成成功")
                    return scene
                else:
                    scene['scene_summary'] = ""
                    scene['smart_read'] = ""
                    print(f"    ✗ 摘要生成失败: 响应格式错误")
                    return scene
            else:
                scene['scene_summary'] = ""
                scene['smart_read'] = ""
                print(f"    ✗ 摘要生成失败: HTTP {response.status_code}")
                return scene
        except Exception as e:
            scene['scene_summary'] = ""
            scene['smart_read'] = ""
            print(f"    ✗ 摘要生成异常: {e}")
            return scene
    
    def summarize_scenes(self, scenes_with_audio: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        批量生成场景摘要
        
        Args:
            scenes_with_audio: 包含音频信息的场景列表
            
        Returns:
            包含摘要的场景列表
        """
        print(f"\n{'='*60}")
        print(f"开始生成场景摘要，共 {len(scenes_with_audio)} 个场景")
        print(f"{'='*60}\n")
        
        scenes_with_summary = []
        
        for i, scene in enumerate(scenes_with_audio):
            scene_id = scene['scene_id']
            print(f"[{i+1}/{len(scenes_with_audio)}] 生成场景 {scene_id} 摘要")
            
            summarized_scene = self.generate_scene_summary(scene)
            scenes_with_summary.append(summarized_scene)
            
            # 打印摘要预览
            scene_summary = summarized_scene.get('scene_summary', '')
            smart_read = summarized_scene.get('smart_read', '')
            if scene_summary and smart_read:
                print(f"    摘要预览: {scene_summary[:100]}...")
                print(f"    智读预览: {smart_read[:100]}...")
            else:
                print(f"    摘要生成失败")
            
            print()
        
        print(f"{'='*60}")
        print(f"场景摘要生成完成，共 {len(scenes_with_summary)} 个场景")
        print(f"{'='*60}\n")
        
        return scenes_with_summary
    
    def generate_video_summary(self, scenes_with_summary: List[Dict[str, Any]], video_path: str) -> Dict[str, Any]:
        """
        生成视频整体摘要
        
        Args:
            scenes_with_summary: 包含摘要的场景列表
            video_path: 视频文件路径
            
        Returns:
            视频整体摘要信息
        """
        print(f"\n{'='*60}")
        print(f"开始生成视频整体摘要")
        print(f"{'='*60}\n")
        
        # 提取所需字段
        transcription_text_list = []
        
        for scene in scenes_with_summary:
            # 提取语音内容
            transcription_text = scene.get('transcription_text', '')
            if transcription_text:
                transcription_text_list.append(transcription_text)
        
        # 构建提示词
        prompt_template = """你是一个专业的视频内容分析AI。请严格按照指定的JSON格式输出分析结果，确保格式正确且内容完整。
        请对以下所有片段的语音内容、视觉内容和摘要内容进行详细总结和分析（片段是按时间顺序出自同一个视频），要求包括：
        1.  视觉总结：理解所有引用的视觉内容，请按片段顺序，连贯、简要地总结整个视频的视觉信息流，字数适中，侧重视觉信息。输出必须是一个字符串数组。
        2.  重点解析：理解所有引用内容，按你理解的视频内容顺序列出若干个关键要点。每个要点必须是"具体主题: 内容"的格式（具体主题和内容在一个字符串中,中间用英文冒号加空格隔开），具体主题要明确具体（如“军事行动展示”、“战争残酷性刻画”等），内容要简要描述。**严禁使用固定的"主题"作为具体主题**，必须使用能够准确概括内容的具体主题名称。所有要点放在一个字符串数组中。
        3.  主要说话人观点：总结不同说话人的主要观点。每个说话人对应一个字符串，格式为"spk_X: 观点总结"（中间用英文冒号加空格隔开）。必须识别所有语音内容中出现的说话人，不能遗漏，如果某个说话人没有实质性内容（如只有语气词），也应列出但注明无实质观点。如果语音内容中没有说话人，则生成空数组[]。
        4.  关键词提取：提取视频中以下几个主题的关键词，严格按照以下顺序，**每个主题对应一个字符串数组，多个关键词分别存入数组（即使只有1个关键词也保持数组格式），组成一个JSON对象**：
        主题顺序：氛围、情感感受、场景、领域、时代、拍摄角度、主题、风格、视角、事件行为、时间段（时间段只能为白天或黑夜）
        5.  总结分析：根据以上所有分析，对视频进行综合性总结。输出必须是一个字符串数组。

        语音内容：
        {transcription_text_list}

        请按照以上要求提供详细的总结分析。

        输出格式要求：
        使用JSON结构化格式，严格遵守JSON语法：
        {{
        "visionSummary": ["视觉总结完整内容"],
        "keyPoints": ["具体主题1: 对该主题的简要描述内容。", 
                      "具体主题2: 对该主题的简要描述内容。",
                      "具体主题3: 对该主题的简要描述内容。",],
        "speakerSummaries": ["spk_0: 对spk_0主要观点的总结。", 
                             "spk_1: 对spk_1主要观点的总结。"],
        "keyWords": {{
            "氛围": ["关键词1", "关键词2"],
            "情感感受": ["关键词1", "关键词2"],
            "场景": ["关键词1", "关键词2"],
            "领域": ["关键词1", "关键词2"],
            "时代": ["关键词1"],
            "拍摄角度": ["关键词1", "关键词2"],
            "主题": ["关键词1", "关键词2"],
            "风格": ["关键词1", "关键词2"],
            "视角": ["关键词1"],
            "事件行为": ["关键词1", "关键词2"],
            "时间段": ["白天或黑夜"]
        }},
        "introSummary": ["总结分析完整内容"]
        }}"""
        
        # 格式化提示词
        formatted_transcription = "\n".join([f"- {text}" for text in transcription_text_list]) if transcription_text_list else "无"
        
        formatted_prompt = prompt_template.format(
            transcription_text_list=formatted_transcription
        )
        
        # 调用视频分析服务
        try:
            # 服务地址
            BASE_URL = "http://192.168.86.113:8000"
            url = f"{BASE_URL}/analyze-video"
            
            # 构造请求数据（按照client.py的格式）
            # 使用绝对路径确保服务能找到文件
            import os
            abs_video_path = os.path.abspath(video_path)
            data = {
                "video_path": abs_video_path,
                "prompt": formatted_prompt,
                "sample_fps": 0.2
            }
            
            print(f"  使用绝对路径: {abs_video_path}")
            print(f"  生成视频整体摘要...")
            print(f"  视频路径: {video_path}")
            print(f"  采样帧率: 0.2")
            
            # 发送请求（使用更长的超时时间）
            response = self.session.post(url, json=data, timeout=600)  # 10分钟超时
            
            # 检查响应状态
            response.raise_for_status()
            
            # 解析响应
            result = response.json()
            
            if 'result' in result:
                video_summary = result['result']
                
                print(f"✓ 视频整体摘要生成成功")
                print(video_summary)
                
                # 解析返回结果
                import re
                video_summary_data = {}

                # 第一步：先去除 ```json 和 ``` 标记，清理格式
                cleaned_video_summary = re.sub(r'```(json)?\n?|\n?```', '', video_summary).strip()

                # 第二步：优先尝试直接解析JSON（推荐，比正则更可靠）
                try:
                    # 直接将清理后的字符串解析为JSON对象
                    json_result = json.loads(cleaned_video_summary)
                    # 从JSON对象中提取字段，直接赋值
                    video_summary_data['visionSummary'] = json_result.get('visionSummary', [])
                    video_summary_data['keyPoints'] = json_result.get('keyPoints', [])
                    video_summary_data['speakerSummaries'] = json_result.get('speakerSummaries', [])
                    video_summary_data['keyWords'] = json_result.get('keyWords', {})  # 修正默认值为对象
                    video_summary_data['introSummary'] = json_result.get('introSummary', [])
                    
                    print(f"\n{'='*60}")
                    print(f"视频整体摘要生成完成")
                    print(f"{'='*60}\n")
                    
                    return video_summary_data
                except json.JSONDecodeError:
                    # 如果JSON解析失败，再用正则兜底提取
                    print("  JSON解析失败，使用正则兜底提取")
                    
                    # 提取视觉总结（匹配 "visionSummary": [内容]，适配JSON格式）
                    vision_match = re.search(r'"visionSummary"\s*:\s*\[(.*?)\]', cleaned_video_summary, re.DOTALL)
                    if vision_match:
                        # 清理内容中的引号、逗号等多余字符
                        vision_content = re.sub(r'["\n,]+', '', vision_match.group(1)).strip()
                        video_summary_data['visionSummary'] = [vision_content] if vision_content else []
                    else:
                        video_summary_data['visionSummary'] = []
                    
                    # 提取重点解析
                    key_points_match = re.search(r'"keyPoints"\s*:\s*\[(.*?)\]', cleaned_video_summary, re.DOTALL)
                    if key_points_match:
                        key_points = [point.strip().replace('"', '') for point in key_points_match.group(1).split(',') if point.strip()]
                        video_summary_data['keyPoints'] = key_points
                    else:
                        video_summary_data['keyPoints'] = []
                    
                    # 提取主要说话人观点
                    speaker_match = re.search(r'"speakerSummaries"\s*:\s*\[(.*?)\]', cleaned_video_summary, re.DOTALL)
                    if speaker_match:
                        speakers = [speaker.strip().replace('"', '') for speaker in speaker_match.group(1).split(',') if speaker.strip()]
                        video_summary_data['speakerSummaries'] = speakers
                    else:
                        video_summary_data['speakerSummaries'] = []
                    
                    # 提取关键词（匹配对象格式）
                    keywords_match = re.search(r'"keyWords"\s*:\s*\{(.*?)\}', cleaned_video_summary, re.DOTALL)
                    if keywords_match:
                        # 尝试解析为JSON对象
                        try:
                            keywords_str = '{"keyWords": {' + keywords_match.group(1) + '}}'
                            keywords_json = json.loads(keywords_str)
                            video_summary_data['keyWords'] = keywords_json.get('keyWords', {})
                        except:
                            # 如果解析失败，返回空对象
                            video_summary_data['keyWords'] = {}
                    else:
                        video_summary_data['keyWords'] = {}
                    
                    # 提取总结分析
                    intro_match = re.search(r'"introSummary"\s*:\s*\[(.*?)\]', cleaned_video_summary, re.DOTALL)
                    if intro_match:
                        intro_content = re.sub(r'["\n,]+', '', intro_match.group(1)).strip()
                        video_summary_data['introSummary'] = [intro_content] if intro_content else []
                    else:
                        video_summary_data['introSummary'] = []
                    
                    print(f"\n{'='*60}")
                    print(f"视频整体摘要生成完成")
                    print(f"{'='*60}\n")
                    
                    return video_summary_data
            else:
                print(f"    ✗ 摘要生成失败: HTTP {response.status_code}")
                return {}
        except Exception as e:
            print(f"    ✗ 摘要生成异常: {e}")
            import traceback
            traceback.print_exc()
            return {}
    
    def generate_mind_map(self, scenes_with_summary: List[Dict[str, Any]]) -> str:
        """
        生成视频思维导图
        
        Args:
            scenes_with_summary: 包含摘要的场景列表
            
        Returns:
            Mermaid语法格式的思维导图
        """
        print(f"\n{'='*60}")
        print(f"开始生成视频思维导图")
        print(f"{'='*60}\n")
        
        # 提取所需字段
        transcription_text_list = []
        scene_summary_list = []
        smart_read_list = []
        
        for scene in scenes_with_summary:
            # 提取语音内容
            transcription_text = scene.get('transcription_text', '')
            if transcription_text:
                transcription_text_list.append(transcription_text)
            
            # 提取视觉内容
            scene_summary = scene.get('scene_summary', scene.get('summary', ''))
            if scene_summary:
                scene_summary_list.append(scene_summary)
            
            # 提取摘要内容
            smart_read = scene.get('segment_insight', scene.get('smart_read', ''))
            if smart_read:
                smart_read_list.append(smart_read)
        
        print(f"  提取到的内容:")
        print(f"  - 语音内容数量: {len(transcription_text_list)}")
        print(f"  - 视觉内容数量: {len(scene_summary_list)}")
        print(f"  - 摘要内容数量: {len(smart_read_list)}")
        
        # 构建提示词
        prompt_template = """你是一个专业的视频内容深度分析 AI。请对提供的视频片段（语音、视觉、摘要）进行全面整合，生成一个结构清晰、层级分明、文字精炼的思维导图数据。
        核心目标：
        生成一个符合特定 Schema 的 JSON 对象。
        中心主题：概括视频主旨（简练有力）。
        动态层级与字数控制（关键）：
        一级分支 (L1)：概括主要逻辑维度，文字适中（约 10-15 字）。
        二级分支 (L2)：详细描述核心内容，文字较丰富（约 20-30 字），可包含背景、现象等。
        三级及以下分支 (L3, L4...)：必须精简文字！
        仅保留核心关键词、关键数据或简短结论。
        字数控制在 10-15 字以内。
        目的：避免深层节点文字过多导致框体过大或排版拥挤，确保图表整体清爽。
        结构策略：
        若内容复杂，通过增加分支数量（横向展开）来承载信息，而不是在一个节点里堆砌长文本。
        例如：将一段长描述拆分为 2-3 个并列的 L3 节点，每个节点只说一个点。
        布局要求：所有分支均位于右侧。不要在 JSON 中包含 left 字段，或确保其值为 false。
        格式与样式规范（重要）：
        输出格式：仅输出标准的 JSON 字符串，不要包含 markdown 代码块标记（如 ```json），不要输出任何解释性文字。
        JSON 结构定义：
        根节点必须包含 name 和 children。
        每个节点对象包含：
        name: (String) 节点文字。
        children: (Array) 子节点数组（可选，若无子节点可省略）。
        collapse: (Boolean) 可选。若该节点有子节点且层级较深（如 L3 及以上），建议设为 true 默认折叠，保持界面整洁。
        注意：不要生成 left 字段，默认所有节点均在右侧展示。
        内容清洗：
        确保 JSON 合法，字符串内的双引号需转义。
        确保层级关系正确，子节点必须嵌套在父节点的 children 数组中。
        参考输出范例：
        {{
        "name": "视频核心主题解析",
        "children": [
        {{
        "name": "主要逻辑维度一",
        "children": [
        {{
        "name": "维度一的详细描述，包含具体背景与核心现象，文字量适中以确保清晰"
        }},
        {{
        "name": "维度一的另一侧面，结合视觉特征的深度解读，避免任何形式的简略概括"
        }}
        ]
        }},
        {{
        "name": "主要逻辑维度二",
        "children": [
        {{
        "name": "维度二的核心议题概述，引出下方的详细拆解分析",
        "children": [
        {{ "name": "具体表现形态一：快速驶过" }},
        {{ "name": "具体表现形态二：紧急停车" }},
        {{ "name": "视觉特征：海上视角实拍" }}
        ]
        }}
        ]
        }},
        {{
        "name": "主要逻辑维度三",
        "collapse": true,
        "children": [
        {{ "name": "第一阶段：起始状态" }},
        {{ "name": "第二阶段：互动关系" }}
        ]
        }}
        ]
        }}
        输入数据：
        语音内容：
        {transcription_text_list}

        视觉内容：
        {scene_summary_list}

        摘要内容：
        {smart_read_list}
        """

        # 格式化提示词
        formatted_transcription = "\n".join([f"- {text}" for text in transcription_text_list]) if transcription_text_list else "无"
        formatted_scene_summary = "\n".join([f"- {summary}" for summary in scene_summary_list]) if scene_summary_list else "无"
        formatted_smart_read = "\n".join([f"- {read}" for read in smart_read_list]) if smart_read_list else "无"
        
        formatted_prompt = prompt_template.format(
            transcription_text_list=formatted_transcription,
            scene_summary_list=formatted_scene_summary,
            smart_read_list=formatted_smart_read
        )
        
        print(f"  提示词长度: {len(formatted_prompt)} 字符")
        
        # 调用LLM模型
        try:
            print(f"  开始调用Qwen-3模型生成思维导图...")
            # 调用glm-5模型生成文本
            mind_map = generate_text_by_qwen3_5(formatted_prompt)
            
            print(f"  模型返回结果长度: {len(mind_map)} 字符")
            
            if mind_map:
                print(f"    ✓ 视频思维导图生成成功")
                print(f"  思维导图内容: {mind_map[:200]}...")  # 只打印前200个字符
                
                # 清理Mermaid代码格式
                import re
                # 去除可能的代码块标记
                cleaned_mind_map = re.sub(r'```(mermaid)?\n?|\n?```', '', mind_map).strip()
                
                print(f"  清理后思维导图长度: {len(cleaned_mind_map)} 字符")
                
                print(f"\n{'='*60}")
                print(f"视频思维导图生成完成")
                print(f"{'='*60}\n")
                
                return cleaned_mind_map
            else:
                print(f"    ✗ 思维导图生成失败: 模型返回空结果")
                return ""
        except Exception as e:
            print(f"    ✗ 思维导图生成异常: {e}")
            import traceback
            traceback.print_exc()
            return ""
    
    def extract_key_frames_from_analysis(self, analyzed_frames: List[Dict[str, Any]], similarity_threshold: float = 0.8) -> List[Dict[str, Any]]:
        """
        从分析后的帧中提取关键帧
        
        关键帧定义：
        1. 检测到face_detections或object_detections的帧
        2. 两帧之间的相似度不超过similarity threshold
        
        Args:
            analyzed_frames: 分析后的帧信息列表
            similarity_threshold: 相似度阈值，超过此值的帧不被视为关键帧
            
        Returns:
            提取的关键帧列表
        """
        if not analyzed_frames:
            return []
        
        key_frames = []
        key_frames_index = []

        # 筛选有检测结果的帧
        detected_frames = []
        for frame in analyzed_frames:
            has_faces = len(frame.get('face_detections', [])) > 0
            has_objects = len(frame.get('object_detections', [])) > 0
            if has_faces or has_objects:
                detected_frames.append(frame)
        
        if not detected_frames:
            return []
        
        # 添加第一帧作为关键帧
        key_frames.append(detected_frames[0])
        key_frames_index.append(detected_frames[0]['frame_index'])
        
        # 遍历剩余帧，计算相似度并筛选
        for i in range(1, len(detected_frames)):
            prev_frame = key_frames[-1]
            curr_frame = detected_frames[i]
            
            similarity = self.compute_weighted_scene_similarity(prev_frame, curr_frame)
            
            if similarity <= similarity_threshold:
                key_frames.append(curr_frame)
                key_frames_index.append(curr_frame['frame_index'])
        
        print(f"从 {len(detected_frames)} 个检测帧中提取出 {len(key_frames)} 个关键帧")
        return key_frames, key_frames_index
    
    def extract_consecutive_items(self, frames: List[Dict[str, Any]], item_key: str) -> List[Dict[str, Any]]:
        """
        提取连续出现超过3帧的物品（人脸或物体）
        
        Args:
            frames: 分析后的帧信息列表
            item_key: 'face_detections' 或 'object_detections'
            
        Returns:
            提取的物品列表，每个物品包含name、box、start_time、end_time
        """
        if not frames:
            return []
        
        tracker = {}
        
        for frame in frames:
            frame_index = frame['frame_index']
            frame_data = frame['frame_data']
            timestamp = frame['timestamp_seconds']
            items = frame.get(item_key, [])
            
            for item in items:
                item_name = item.get('name', '')
                item_box = item.get('box', [])
                
                if not item_name:
                    continue
                
                if item_name not in tracker:
                    tracker[item_name] = [{
                        'current_streak': 1,
                        'frame_data': frame_data,
                        'start_frame': frame_index,
                        'end_frame': frame_index,
                        'box': item_box,
                        'start_time': timestamp,
                        'end_time': timestamp
                    }]
                else:
                    last_streak = tracker[item_name][-1]
                    if frame_index == last_streak['end_frame'] + self.frame_interval:
                        last_streak['current_streak'] += 1
                        last_streak['end_frame'] = frame_index
                        last_streak['end_time'] = timestamp
                    else:
                        tracker[item_name].append({
                            'current_streak': 1,
                            'frame_data': frame_data,
                            'start_frame': frame_index,
                            'end_frame': frame_index,
                            'box': item_box,
                            'start_time': timestamp,
                            'end_time': timestamp
                        })
        
        result = []
        for item_name, streaks in tracker.items():
            tagTrack = []
            valid_streaks = []
            
            # 收集所有连续出现超过3帧的streak
            for streak in streaks:
                if streak['current_streak'] >= 3:
                    valid_streaks.append(streak)
                    tagTrack.append({
                        "startTime": streak['start_time'],
                        "endTime": streak['end_time']
                    })
            
            # 对于同一个物品，只添加一个记录，包含所有时间段
            if valid_streaks:
                # 使用第一个有效streak的信息作为基础
                first_streak = valid_streaks[0]
                result.append({
                    'name': item_name,
                    'box': first_streak['box'],
                    'start_frame': first_streak['start_frame'],
                    'frame_data': first_streak['frame_data'],
                    'tagTrack': tagTrack  # 包含所有时间段
                })
        
        return result
    
    def save_results(self, analyzed_frames: List[Dict[str, Any]], scenes_with_summary: List[Dict[str, Any]], video_summary: Dict[str, Any], mind_map: str, audio_results: Dict[str, Any], video_path: str) -> str:
        """
        保存分析结果为JSON文件
        
        Args:
            analyzed_frames: 分析后的帧信息列表
            scenes_with_summary: 包含摘要的场景信息列表
            video_summary: 视频摘要
            video_path: 视频文件路径
            
        Returns:
            保存的JSON文件路径
        """
        print(f"\n{'='*60}")
        print(f"保存分析结果")
        print(f"{'='*60}")

        sceneSegments = []
        tags = []
        visionSummary = ""
        introSummary = ""
        keyPoints = []
        speakerSummaries = []
        mindMap = mind_map
        audioSegments = []
        textVectors = []


        vision_summary_list = video_summary.get('visionSummary', [])
        keyPoints = video_summary.get('keyPoints', [])
        spSumms = video_summary.get('speakerSummaries', [])
        keyWords = video_summary.get('keyWords', {})
        intro_summary_list = video_summary.get('introSummary', [])

        # 提取关键帧
        # key_frames, key_frames_index = self.extract_key_frames_from_analysis(analyzed_frames, similarity_threshold=0.6)
        
        # 提取连续出现超过3帧的人脸和物体
        faces = self.extract_consecutive_items(analyzed_frames, 'face_detections')
        objects = self.extract_consecutive_items(analyzed_frames, 'object_detections')
        
        # 不再创建本地目录，直接上传到RustFS
        
        video_name = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(video_name)[0]
        
        for i, face in enumerate(faces):
            start_frame = face['start_frame']
            image_frame = face['frame_data']
            box = face['box']
            face_name = face['name']
            # face_vector = self.face_embeddings.get(face_name, None)
            
            # 查找对应帧的图像数据
            # image_frame = None
            # for frame in analyzed_frames:
            #     if frame['frame_index'] == start_frame:
            #         image_frame = frame.get('frame_data')
            #         break
            
            if image_frame is not None:
                # 切图
                cropped_face = self.crop_image(image_frame, box)
                if cropped_face is not None:
                    # 直接上传到RustFS，不保存到本地
                    object_name = f"catalog_data_v3/resource_{self.resource_id}/face_image/{face_name}_{start_frame}.jpg"
                    url_image_face = asyncio.run(self.upload_image_to_rustfs(cropped_face, object_name, is_path=False))
                    
                    tags.append({
                        "tagCode": "face",
                        "tagValue": face_name,
                        "tagTrack": face['tagTrack'],
                        "imageUrl": url_image_face,
                        "vector": []
                    })
        
        for i, object in enumerate(objects):
            start_frame = object['start_frame']
            image_frame = object['frame_data']
            box = object['box']
            obj_name = object['name']
            obj_value = self.object_map.get(obj_name, obj_name)
            
            # 查找对应帧的图像数据
            # image_frame = None
            # for frame in analyzed_frames:
            #     if frame['frame_index'] == start_frame:
            #         image_frame = frame.get('frame_data')
            #         break
            
            if image_frame is not None:
                # 直接上传到RustFS，不保存到本地
                object_name = f"catalog_data_v3/resource_{self.resource_id}/object_image/{obj_name}_{start_frame}.jpg"
                url_image_object = asyncio.run(self.upload_image_to_rustfs(image_frame, object_name, is_path=False))
                
                tags.append({
                    "tagCode": obj_name,
                    "tagValue": obj_value,
                    "tagTrack": object['tagTrack'],
                    "imageUrl": url_image_object,
                    "vector": []
                })
        
        for scene in scenes_with_summary:
            scene_id = scene['scene_id']
            start_frame = scene['start_frame']
            frame_data = None
            for frame in analyzed_frames:
                if frame['frame_index'] == start_frame:
                    frame_data = frame.get('frame_data')
                    break

            if frame_data is not None:
                # 直接上传到RustFS，不保存到本地
                object_name = f"catalog_data_v3/resource_{self.resource_id}/scene_image/scene_{scene_id}.jpg"
                url_image_scene = asyncio.run(self.upload_image_to_rustfs(frame_data, object_name, is_path=False))
                print(f"  上传场景图像到RustFS: {url_image_scene}")
            
            sceneSegments.append({
                "sceneId": scene_id,
                "startTime": scene['start_time'],
                "endTime": scene['end_time'],
                "duration": scene['duration'],
                "imageUrl": url_image_scene,
                "sceneSummary": scene['scene_summary'],
                "smartRead": scene['smart_read']
            })
        
        for summary in vision_summary_list:
            visionSummary += summary
        
        for summary in intro_summary_list:
            introSummary += summary

        for key, values in keyWords.items():
            tag_code = self.keyword_code.get(key)
            for value in values:
                tags.append({
                    "tagCode": tag_code,
                    "tagValue": value,
                    "tagTrack": [],
                    "imageUrl": "",
                    "vector": []
                })
        
        for i, spSumm in enumerate(spSumms):
            text = spSumm.split(": ")[1]
            speakerSummaries.append({
                "speaker": f"发言人{i}",
                "summary": text
            })
        
        for audio in audio_results:
            audioSegments.append({
                "startTime": audio['start_time'],
                "endTime": audio['end_time'],
                "audioText": audio['text'],
            })

        # 准备保存数据
        results = {
            "sceneSegments": sceneSegments,
            "tags": tags,
            "visionSummary": visionSummary,
            "introSummary": introSummary,
            "keyPoints": keyPoints,
            "speakerSummaries": speakerSummaries,
            "mindMap": mindMap,
            "audioSegments": audioSegments,
            "textVectors": textVectors,
        }
        
        # 生成输出文件路径
        results_path = "/data_ssd/smart_catalog_release_v1.1"
        video_name = os.path.basename(video_path)
        video_name_without_ext = os.path.splitext(video_name)[0]
        output_dir = os.path.join(results_path, "catalog_results", "analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, f"{video_name_without_ext}_analysis.json")
        
        try:
            # 保存为JSON文件
            with open(output_path, 'w', encoding='utf-8') as f:
                # 处理numpy数组等不可序列化的对象
                def serialize(obj):
                    if isinstance(obj, np.ndarray):
                        return "<numpy array>"
                    raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
                
                json.dump(results, f, ensure_ascii=False, indent=2, default=serialize)
            
            print(f"  ✓ 结果保存成功: {output_path}")
            print(f"  - 帧数: {len(analyzed_frames)}")
            print(f"  - 场景数: {len(scenes_with_summary)}")
            
            return results
        except Exception as e:
            print(f"  ✗ 结果保存失败: {e}")
            return ""
    
    def analyze_key_frames(self, all_frames: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        print(f"\n开始分析 {len(all_frames)} 个视频帧...")
        
        # 读取 face_embedding.json 文件并加载到 self.face_embeddings
        face_embedding_path = "./configs/face_embedding.json"
        try:
            with open(face_embedding_path, 'r', encoding='utf-8') as f:
                face_embeddings_data = json.load(f)
                for item in face_embeddings_data:
                    self.face_embeddings.append(item)
                print(f"  加载了 {len(face_embeddings_data)} 个人脸嵌入数据")
        except Exception as e:
            print(f"  加载人脸嵌入数据失败: {e}")
        
        for i, frame_info in enumerate(all_frames, 1):
            print(f"\n[{i}/{len(all_frames)}] 分析帧 {frame_info['frame_index']}")
            
            image_frame = frame_info['frame_data']
            frame_b64 = self.image_to_base64(frame_info['frame_data'])
            if not frame_b64:
                print(f"  跳过：无法编码图像")
                continue
            
            '''人脸检测与识别'''
            face_result = self.detect_faces(frame_b64)
            if face_result:
                detections = face_result.get('detections', [])
                landmarks = face_result.get('landmarks', [])
                print(f"  人脸检测: 发现 {len(detections)} 个人脸")
                
                # 收集符合面积要求的人脸
                valid_faces = []
                for i, detection in enumerate(detections):
                    [x1, y1, x2, y2] = detection[:4]
                    width = x2 - x1
                    height = y2 - y1
                    area = width * height
                    if area >= self.minface:
                        valid_faces.append({
                            'detection': detection,
                            'landmark': landmarks[i],
                            'box': [int(x1), int(y1), int(x2), int(y2)],
                            'area': area
                        })
                
                # 按面积从大到小排序，只保留前5个主要人脸
                valid_faces.sort(key=lambda x: x['area'], reverse=True)
                main_faces = valid_faces[:5]  # 只保留前5个最大的人脸
                
                print(f"  过滤后: 保留 {len(main_faces)} 个主要人脸")
                
                for j, face_info in enumerate(main_faces):
                    box = face_info['box']
                    landmarks = face_info['landmark']
                    
                    # 根据检测框裁剪人脸区域（添加一些边距）
                    h, w = image_frame.shape[:2]
                    x1, y1, x2, y2 = box
                    margin = 0.2  # 20% 的边距
                    box_w = x2 - x1
                    box_h = y2 - y1
                    x1_new = max(0, x1 - int(margin * box_w))
                    y1_new = max(0, y1 - int(margin * box_h))
                    x2_new = min(w, x2 + int(margin * box_w))
                    y2_new = min(h, y2 + int(margin * box_h))
                    
                    cropped_img = image_frame[y1_new:y2_new, x1_new:x2_new]
                    
                    # 调整关键点坐标到裁剪后的坐标系
                    landmarks_crop = landmarks.copy()
                    for i in range(0, len(landmarks_crop), 2):
                        landmarks_crop[i] -= x1_new     # x 坐标
                        landmarks_crop[i+1] -= y1_new   # y 坐标
                    
                    # 解析关键点
                    left_eye = np.array([landmarks_crop[0], landmarks_crop[1]])
                    right_eye = np.array([landmarks_crop[2], landmarks_crop[3]])
                    nose = np.array([landmarks_crop[4], landmarks_crop[5]])
                    left_mouth = np.array([landmarks_crop[6], landmarks_crop[7]])
                    right_mouth = np.array([landmarks_crop[8], landmarks_crop[9]])
                    
                    # 计算双眼连线与水平线的夹角
                    dy = right_eye[1] - left_eye[1]
                    dx = right_eye[0] - left_eye[0]
                    angle = np.degrees(np.arctan2(dy, dx))
                    
                    # 计算旋转中心（双眼中心）
                    center = ((left_eye[0] + right_eye[0]) / 2, 
                              (left_eye[1] + right_eye[1]) / 2)
                    
                    # 获取旋转矩阵
                    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
                    
                    # 应用旋转变换
                    rotated_img = cv2.warpAffine(cropped_img, rotation_matrix, 
                                                (cropped_img.shape[1], cropped_img.shape[0]), 
                                                flags=cv2.INTER_CUBIC, 
                                                borderMode=cv2.BORDER_REPLICATE)
                    
                    # 定义标准人脸关键点位置（用于仿射变换）
                    target_size = 112  # 模型输入尺寸
                    dst_points = np.array([ 
                        [38.2946, 51.6963],   # 左眼
                        [73.5318, 51.5244],   # 右眼
                        [56.0252, 71.7366],   # 鼻子
                        [41.5493, 92.3655],   # 左嘴角
                        [70.7299, 92.2041]    # 右嘴角
                    ], dtype=np.float32)
                    
                    # 旋转后的关键点位置
                    src_points = np.array([ 
                        left_eye,
                        right_eye,
                        nose,
                        left_mouth,
                        right_mouth
                    ], dtype=np.float32)
                    
                    # 对源关键点应用相同的旋转变换
                    for i in range(len(src_points)):
                        point = src_points[i]
                        transformed_point = np.dot(rotation_matrix[:2], np.array([point[0], point[1], 1]))
                        src_points[i] = transformed_point
                    
                    # 使用 5 点拟合计算仿射变换矩阵
                    affine_matrix, _ = cv2.estimateAffine2D(src_points, dst_points)
                    
                    # 应用仿射变换进行人脸矫正和裁剪
                    aligned_face = cv2.warpAffine(rotated_img, affine_matrix, 
                                               (target_size, target_size), 
                                               flags=cv2.INTER_CUBIC, 
                                               borderMode=cv2.BORDER_REPLICATE)
                    
                    face_b64 = self.image_to_base64(aligned_face)
                    if not face_b64:
                        print(f"    人脸 {j+1}: 无法编码")
                        continue
                    
                    recognize_result = self.recognize_faces(face_b64)
                    embedding = recognize_result.get('embedding', [])
                    if embedding:
                        person_name = self.match_face_embedding(embedding)
                        print(f"    人脸 {j+1}: 匹配到人员 {person_name} (面积: {face_info['area']:.0f})")
                        
                        frame_info['face_detections'].append({
                            "name": person_name,
                            "box": box,
                        })
            else:
                print(f"  人脸检测: 未检测到人脸")
                frame_info['face_detections'] = []
        
            '''目标检测'''
            object_result = self.detect_objects(frame_b64)
            labels = object_result.get('labels', [])
            boxes = object_result.get('boxes', [])
            scores = object_result.get('scores', [])

            valid_objects = []

            if labels and boxes and scores:
                for j, label in enumerate(labels):
                    class_name = labels[j]
                    bbox = boxes[j]
                    score = scores[j]
                    if class_name not in valid_objects:
                        valid_objects.append(class_name)
                        x1, y1, x2, y2 = map(int, bbox[:4])
                        box = [int(x1), int(y1), int(x2), int(y2)]
                        
                        frame_info['object_detections'].append({
                            "name": class_name,
                            "box": box,
                            "score": score,
                        })
                    else:
                        print(f"    目标 {j+1}: 重复检测到 {class_name},不计入结果")
            else:
                print(f"  目标检测: 未检测到物体")
                frame_info['object_detections'] = []
            

            # '''VL模型检测'''
            # if i % 10 == 0 or i == 1:
            #     print(f"  VL模型分析: 进行场景描述")
            #     vl_prompt = "请简要描述这个图像中的内容"
            #     vl_result = self.analyze_scene(frame_b64, vl_prompt)
            #     if vl_result:
            #         frame_info['scene_description'] = vl_result
            #         print(f"场景描述: {frame_info['scene_description'][:100]}...")
            #     else:
            #         frame_info['scene_description'] = ""
            #         print(f"    VL模型分析失败")
            # else:
            #     frame_info['scene_description'] = ""
            
            '''关键帧判断与收集'''
            # 判断是否为关键帧：检测到人脸或物体
            is_key_frame = False
            if frame_info.get('face_detections') and len(frame_info['face_detections']) > 0:
                is_key_frame = True
                print(f"  标记为关键帧: 检测到 {len(frame_info['face_detections'])} 个人脸")
            elif frame_info.get('object_detections') and len(frame_info['object_detections']) > 0:
                is_key_frame = True
                print(f"  标记为关键帧: 检测到 {len(frame_info['object_detections'])} 个物体")
            
            # 如果是关键帧，检查与已有关键帧的相似度
            if is_key_frame:
                # 设置相似度阈值，超过此阈值的帧被认为太相似，不加入关键帧列表
                similarity_threshold = 0.6
                
                # 检查与最近几个关键帧的相似度
                is_similar = False
                recent_key_frames = self.key_frames[-3:]  # 只检查最近3个关键帧，提高效率
                
                for key_frame in recent_key_frames:
                    try:
                        # 计算当前帧与关键帧的相似度
                        similarity = self.calculate_frame_similarity(frame_info['frame_data'], key_frame['frame_data'])
                        print(f"  与关键帧 {key_frame['frame_index']} 的相似度: {similarity:.3f}")
                        
                        if similarity > similarity_threshold:
                            is_similar = True
                            print(f"  过滤掉相似度高的帧")
                            break
                    except Exception as e:
                        print(f"  计算相似度时出错: {e}")
                        continue
                
                # 如果与已有关键帧不相似，则添加到关键帧列表
                if not is_similar:
                    self.key_frames.append(frame_info)
                    print(f"  添加到关键帧列表，当前关键帧数量: {len(self.key_frames)}")
                    
                    # 当累积到10帧时，推送关键帧并清空列表
                    if len(self.key_frames) >= 10:
                        print(f"\n关键帧数量达到10帧，开始推送...")
                        # 运行异步推送任务
                        import asyncio
                        push_success = asyncio.run(self.push_key_frames_to_api(self.task_id, self.resource_id))
                        print(f"关键帧推送 {'成功' if push_success else '失败'}")
                        # 推送成功后清空关键帧列表
                        if push_success:
                            self.key_frames = []
                            print("关键帧列表已清空，继续收集")
        
        # 推送剩余的关键帧数据到API端点
        if self.task_id and self.key_frames:
            print(f"\n视频分析结束，推送剩余的 {len(self.key_frames)} 个关键帧...")
            # 运行异步推送任务
            import asyncio
            push_success = asyncio.run(self.push_key_frames_to_api(self.task_id, self.resource_id))
            print(f"关键帧推送 {'成功' if push_success else '失败'}")
            # 推送成功后清空关键帧列表
            if push_success:
                self.key_frames = []
                print("关键帧列表已清空")
        
        return all_frames


def generate_analysis_file(video_path = "", frame_interval: int = 25, ffmpeg_path = None, task_id = None, resource_id = None):

    import sys
    
    # 如果没有传入ffmpeg_path，尝试从命令行参数获取（仅当直接从命令行调用时）
    if ffmpeg_path is None and __name__ == "__main__" and len(sys.argv) > 2:
        ffmpeg_path = sys.argv[2]
    
    analyzer = VideoFrameAnalyzer(ffmpeg_path=ffmpeg_path, frame_interval=frame_interval, task_id=task_id, resource_id=resource_id)
    
    # if len(sys.argv) > 1:
    #     video_path = sys.argv[1]
    # else:
    #     video_path = "/data_ssd/auto_catalog/uploads/test_1.mp4"
    
    if not os.path.exists(video_path):
        print(f"错误: 视频文件不存在: {video_path}")
        return
    
    try:
        print(f"开始分析视频: {video_path}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        # 平均抽帧
        all_frames = analyzer.extract_key_frames(video_path, analyzer.frame_interval)
        # CV模型分析, VL模型分析
        analyzed_frames = analyzer.analyze_key_frames(all_frames)
        # 场景划分
        scenes = analyzer.segment_scenes(analyzed_frames, similarity_threshold=0.7)
        # 处理场景音频和语音识别
        audio_results, scenes_with_audio = analyzer.process_scenes_audio(video_path, scenes)
        # 生成场景摘要
        scenes_with_summary = analyzer.summarize_scenes(scenes_with_audio)
        # 生成视频总结
        video_summary = analyzer.generate_video_summary(scenes_with_summary, video_path)
        # 生成思维导图
        mind_map = analyzer.generate_mind_map(scenes_with_summary)
        # 保存分析结果
        results = analyzer.save_results(analyzed_frames, scenes_with_summary, video_summary, mind_map, audio_results, video_path)
        
        elapsed_time = time.time() - start_time
        print(f"\n总耗时: {elapsed_time:.2f} 秒")
        return results
        
    except KeyboardInterrupt:
        print("\n用户中断操作")
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


# if __name__ == "__main__":
#     import sys
    
#     # 解析命令行参数
#     if len(sys.argv) < 2:
#         print("使用方法:")
#         print("  python auto_catalog_dep.py <视频文件路径> [ffmpeg路径]")
#         print("示例:")
#         print("  python auto_catalog_dep.py /path/to/video.mp4")
#         print("  python auto_catalog_dep.py /path/to/video.mp4 /usr/bin/ffmpeg")
#         sys.exit(1)
    
#     # 获取视频路径
#     video_path = sys.argv[1]
    
#     # 获取ffmpeg路径（如果提供）
#     ffmpeg_path = None
#     if len(sys.argv) > 2:
#         ffmpeg_path = sys.argv[2]
    
#     # 调用分析函数
#     print(f"开始分析视频: {video_path}")
#     if ffmpeg_path:
#         print(f"使用FFmpeg路径: {ffmpeg_path}")
    
#     # 调用分析函数，传递视频路径和ffmpeg路径
#     results = generate_analysis_file(video_path, ffmpeg_path=ffmpeg_path)
    
#     # 输出结果路径
#     if results:
#         print(f"分析完成，结果已保存到: {results}")
#     else:
#         print("分析失败")
