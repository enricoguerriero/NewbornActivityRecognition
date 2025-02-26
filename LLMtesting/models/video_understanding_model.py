import abc
import torch
import cv2
from tqdm import tqdm
import os
from utils.config import CONFIG

class VideoUnderstandingModel(abc.ABC):
    """
    A generic interface for image understanding models.
    Subclasses should implement the answer_question method.
    """
    def __init__(self, device=None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abc.abstractmethod
    def answer_question(self, video, question: str, seed: int = 42,
                        top_p: float = 0.95, temperature: float = 0.1) -> str:
        """
        Given a video and a question, generate an answer. Specifically, this method should force the model to return 0 or 1.
        """
        pass
    

    def test_model(self, video_folder, questions):
        """
        from each video folder, ask a question to the model and check if this matches the label.
        """
        