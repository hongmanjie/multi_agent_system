# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
from typing import Any, Dict

import cv2
import numpy as np
import PIL
import torch

from .torchkit.backbone.arcface_backbone import _iresnet


class ArcFaceRecognition():
    def __init__(self, **kwargs):
        model_path = kwargs.get("model_path", "")
        self.device = kwargs.get("device", "cpu")
        enable_cuda = kwargs.get("enable_cuda", True)
        self.device = self.device if enable_cuda and torch.cuda.is_available() else "cpu"
        # 默认使用 iresnet50 模型
        face_model = _iresnet('arcface_i50', [3, 4, 14, 3])
        face_model.load_state_dict(
            torch.load(
                model_path,
                map_location=self.device))
        face_model = face_model.to(self.device)
        face_model.eval()
        self.face_model = face_model

    def preprocess(self, input) -> Dict[str, Any]:
        result = {}
        if input is None:
            result['img'] = None
            return result
        
        resized_img = cv2.resize(input, (112, 112))

        # face_img = resized_img[:, :, ::-1]  # to rgb
        # face_img = np.transpose(face_img, axes=(2, 0, 1))
        # face_img = (face_img / 255. - 0.5) / 0.5
        # face_img = face_img.astype(np.float32)
        # result['img'] = torch.from_numpy(face_img).to(self.device)

        img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
        img = np.transpose(img, (2, 0, 1))
        img = torch.from_numpy(img).unsqueeze(0).float()
        img.div_(255).sub_(0.5).div_(0.5)
        img = img.to(self.device)
        result['img'] = img
        return result

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        if input['img'] is None:
            return None
        # img = input['img'].unsqueeze(0)
        img = input['img']
        emb = self.face_model(img).detach().cpu().numpy()
        emb /= np.sqrt(np.sum(emb**2, -1, keepdims=True))  # l2 norm
        return emb

    
    def predict(self, image):
        image = np.array(image)
        result = self.preprocess(image)
        forward_result = self.forward(result) #(1, 512)

        result = {}
        rec_result = forward_result[0].tolist() if isinstance(forward_result, np.ndarray) else forward_result
        result["embedding"] = rec_result
        return result
