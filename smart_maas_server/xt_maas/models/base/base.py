from abc import ABC, abstractmethod

class CV_BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        # 统一处理device
        pass
    
    abstractmethod
    def predict(self, *args, **kwargs):
        pass

class MultiModal_BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    abstractmethod
    def generate(self, *args, **kwargs):
        pass

class NLP_BaseModel(ABC):
    def __init__(self, *args, **kwargs):
        pass
    
    abstractmethod
    def generate(self, *args, **kwargs):
        pass