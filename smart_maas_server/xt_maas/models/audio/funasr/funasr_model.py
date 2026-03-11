from funasr import AutoModel
import numpy as np
import tempfile
import os
from typing import List, Dict, Any
from datetime import datetime


class FunASRModel:
    def __init__(self, **kwargs):
        """
        初始化FunASR模型
        
        Args:
            **kwargs: 模型配置参数
        """
        self.model_config = kwargs
        self.model = None
        self.is_loaded = False
        # 在初始化时加载模型，而不是在调用时加载
        self.load_model()
    
    def load_model(self):
        """加载FunASR模型"""
        try:
            print("正在加载FunASR模型...")
            self.model = AutoModel(
                model=self.model_config.get("model", "paraformer-zh"),
                vad_model=self.model_config.get("vad_model", "fsmn-vad"), 
                spk_model=self.model_config.get("spk_model", "cam++"),
                punc_model=self.model_config.get("punc_model", "ct-punc-c"),
                device=self.model_config.get("device", "cuda:3")
            )
            self.is_loaded = True
            print("FunASR模型加载完成!")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def transcribe_audio(self, audio_data: bytes, audio_format: str = "wav") -> Dict[str, Any]:
        """
        语音转文字处理
        
        Args:
            audio_data: 音频数据
            audio_format: 音频格式
            
        Returns:
            语音识别结果
        """
        if not self.is_loaded:
            self.load_model()
        
        try:
            # 保存临时音频文件
            with tempfile.NamedTemporaryFile(suffix=f".{audio_format}", delete=False) as temp_audio:
                temp_audio.write(audio_data)
                temp_audio_path = temp_audio.name
            
            # 使用FunASR进行语音识别
            result = self.model.generate(
                input=temp_audio_path,
                batch_size_s=300,  # 批处理大小
                hotword='',  # 热词
            )
            
            # 清理临时文件
            os.unlink(temp_audio_path)
            
            return self._parse_result(result)
            
        except Exception as e:
            print(f"语音识别错误: {e}")
            raise
    
    def _parse_result(self, result: List[Dict]) -> Dict[str, Any]:
        """
        解析FunASR返回结果
        
        Args:
            result: FunASR返回的原始结果
            
        Returns:
            解析后的结果
        """
        if not result:
            return {"success": False, "message": "未识别到语音", "results": []}
        
        transcription_results = []
        
        for res in result:
            if 'sentence_info' in res:
                # 多说话人场景
                for sentence_info in res['sentence_info']:
                    speaker_id = sentence_info.get('spk', 'unknown')
                    text = sentence_info.get('text', '')
                    start_time = sentence_info.get('start', 0) / 1000
                    end_time = sentence_info.get('end', 0) / 1000
                    
                    transcription_results.append({
                        "speaker_id": f"spk_{speaker_id}",
                        "text": text,
                        "start_time": start_time,
                        "end_time": end_time,
                        "timestamp": f"{start_time:.2f}-{end_time:.2f}s"
                    })
            else:
                # 单说话人场景
                text = res.get('text', '')
                if text:
                    transcription_results.append({
                        "speaker_id": "spk_0",
                        "text": text,
                        "start_time": 0,
                        "end_time": 0,
                        "timestamp": "unknown"
                    })
        
        return {
            "success": True,
            "message": "识别成功",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "total_speakers": len(set([r["speaker_id"] for r in transcription_results])),
            "total_segments": len(transcription_results),
            "results": transcription_results
        }
    
    def predict(self, audio_data, audio_format: str = "wav") -> Dict[str, Any]:
        """
        预测接口，用于与smart_maas_server框架集成
        
        Args:
            audio_data: 音频数据
            audio_format: 音频格式
            
        Returns:
            预测结果
        """
        try:
            if not audio_data:
                return {"success": False, "message": "缺少音频数据"}
            
            result = self.transcribe_audio(audio_data, audio_format)
            return result
        except Exception as e:
            return {"success": False, "message": f"处理失败: {str(e)}"}
