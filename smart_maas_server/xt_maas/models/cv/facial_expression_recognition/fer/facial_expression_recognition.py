# The implementation is based on Facial-Expression-Recognition, available at
# https://github.com/WuJie1010/Facial-Expression-Recognition.Pytorch
import os

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from PIL import Image
from torch.autograd import Variable

from . import transforms
from .vgg import VGG
from xt_maas.utils.config import load_config


class FacialExpressionRecognition():

    def __init__(self, **kwargs):
        cudnn.benchmark = True
        config_path = kwargs.get("config_path", "")
        self.device = kwargs.get("device", "cpu")
        enable_cuda = kwargs.get("enable_cuda", True)
        self.device = self.device if enable_cuda and torch.cuda.is_available() else "cpu"

        config = load_config(config_path)
        self.cfg = config['models']
        self.model_path = os.path.join(os.path.dirname(config_path), 'pytorch_model.pt')
        self.labels = list(config['labels'])

        self.net = VGG('VGG19', cfg=self.cfg)
        self.load_model()
        self.net = self.net.to(self.device)
        self.transform_test = transforms.Compose([
            transforms.TenCrop(44),
            transforms.Lambda(lambda crops: torch.stack(
                [transforms.ToTensor()(crop) for crop in crops])),
        ])

        self.mean = np.array([[104, 117, 123]])

    def load_model(self):
        pretrained_dict = torch.load(
            self.model_path, map_location=self.device)
        self.net.load_state_dict(pretrained_dict['net'], strict=True)
        self.net.eval()

    def predict(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        img = cv2.resize(img, (48, 48))
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)

        img = Image.fromarray(np.uint8(img))
        inputs = self.transform_test(img)

        ncrops, c, h, w = inputs.shape

        inputs = inputs.view(-1, c, h, w)
        inputs = inputs.to(self.device)
        inputs = Variable(inputs)
        outputs = self.net(inputs)

        outputs_avg = outputs.view(ncrops, -1).mean(0)  # avg over crops

        score = F.softmax(outputs_avg, dim=0)
        _, predicted = torch.max(outputs_avg.data, 0)
        # print(score)
        result = {}
        result['score'] = score.detach().cpu().numpy().tolist()
        result['predicted'] = self.labels[int(predicted.item())]
        # return score, predicted
        return result
