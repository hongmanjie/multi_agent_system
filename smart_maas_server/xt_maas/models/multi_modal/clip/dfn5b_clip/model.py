import os
import re
import json
import torch
import open_clip
from PIL import Image
from transformers import MarianMTModel, MarianTokenizer
import numpy as np

def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True

def replace_keywords(text, keyword_dict):
    """
    根据JSON文件中的关键词对应表替换文本中的关键词
    :param text: 输入文本（字符串）
    :param keyword_dict: 存储替换规则dict
    :return: 替换后的文本
    """
    if not keyword_dict:
        return text
    
    # 生成正则表达式模式，匹配字典中所有键（转义特殊字符）
    pattern = re.compile(r'(' + '|'.join(map(re.escape, keyword_dict.keys())) + r')')
    
    # 使用回调函数进行替换
    replaced_text = pattern.sub(lambda x: keyword_dict[x.group()], text)
    return replaced_text

class DFN5B_CLIP:
    def __init__(self, model_type='ViT-H-14', 
                clip_path='hf-hub:apple/DFN5B-CLIP-ViT-H-14',
                opus_path='hf-hub:apple/DFN5B-Opus',
                keyword_map_json='keyword_map.json', 
                device="cuda:0",
                enable_cuda=True):
        self.device = torch.device(device if enable_cuda and torch.cuda.is_available() else "cpu")
        
        self.clip_model_name = 'local-dir:' + clip_path

        # 加载CLIP模型和processor
        self.model_clip, self.processor_clip = open_clip.create_model_from_pretrained(model_name=self.clip_model_name)
        self.model_tokenizer = open_clip.get_tokenizer(model_name=self.clip_model_name)
        self.model_clip = self.model_clip.to(self.device)
        print('OPENCLIP MODEL LOAD SUCCESS!')

        # 加载关键词映射字典
        self.keyword_dict = {}
        if os.path.exists(keyword_map_json):
            with open(keyword_map_json, 'r', encoding='utf-8') as f:
                self.keyword_dict = json.load(f)

        # 加载MarianMT翻译模型（中文 -> 英文）
        self.opus_model_name = opus_path
        self.marian_tokenizer = MarianTokenizer.from_pretrained(pretrained_model_name_or_path=self.opus_model_name)
        self.marian_model = MarianMTModel.from_pretrained(pretrained_model_name_or_path=self.opus_model_name)
        self.marian_model = self.marian_model.to(self.device)  

        print('MARIAN TRANSLATION MODEL LOAD SUCCESS!')


    def translate_zh_to_en(self, text_zh):
        inputs = self.marian_tokenizer(text_zh, return_tensors="pt", padding=True, truncation=True).to(self.device)
        outputs = self.marian_model.generate(**inputs)
        translated_text = self.marian_tokenizer.decode(outputs[0], skip_special_tokens=True)
        return translated_text

    def encode_input(self, image):
        with torch.no_grad():
        
            if self.device.type == 'cuda':
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    image_tensor = self.processor_clip(image).unsqueeze(0).to(self.device)
                    feature = self.model_clip.encode_image(image_tensor)
                    feature = feature.detach().cpu().numpy()[0].tolist()
            else:
                image_tensor = self.processor_clip(image).unsqueeze(0).to(self.device)
                feature = self.model_clip.encode_image(image_tensor)
                feature = feature.detach().cpu().numpy()[0].tolist()
                
            return feature

    def generate(self, image: Image = None, text: str = None):
        
        text_feat = None
        image_feat = None
        if text:
            if is_all_chinese(text):
                input_sequence = replace_keywords(text, self.keyword_dict)
                text = self.translate_zh_to_en(input_sequence)
            
            # text_token = self.model_tokenizer(text).to(self.device)
            # text_features = self.model_clip.encode_text(text_token)
            # text_feat = text_features.detach().cpu().numpy()[0].tolist()
            text_inputs = self.model_tokenizer(text)  # shape: [1, seq_len], 在 CPU
            # 将其移动到与模型相同的设备，例如 cuda:3
            text_inputs = text_inputs.to(self.device)
            # 编码文本特征
            text_features = self.model_clip.encode_text(text_inputs)
            text_feat = text_features.detach().cpu().numpy()[0].tolist()
        if image:
            image_feat = self.encode_input(image)

        result = {}
        result["text_feat"] = text_feat
        result["image_feat"] = image_feat
        return result 
