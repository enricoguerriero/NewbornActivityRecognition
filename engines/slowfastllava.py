import torch
import torch.nn as nn
import re
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from pytorchvideo.models.hub import slowfast_r50
from engines.prompt_engine import PromptEngine

class SlowFastLlavaEngine(PromptEngine):
    """
    A wrapper that replaces the original visual encoder in Video LLaVA with SlowFast.
    """
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf", device=None):
        super().__init__()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load LLaVA processor and language model
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16
            ).to(self.device)

        self.model.eval()

        # Replace visual encoder with SlowFast
        self.visual_encoder = slowfast_r50(pretrained=True).to(self.device).eval()
        
        # Projection to expected embedding size (768 or 1024 etc.)
        self.proj = nn.Linear(2304, 768).to(self.device)  # Adjust input/output dims as needed

        self.name = "slowfast_llava"

    def extract_features(self, video_tensor: torch.Tensor):
        """
        Apply SlowFast model to a [B, C, T, H, W] tensor video and project features.
        """
        with torch.no_grad():
            feats = self.visual_encoder(video_tensor)  # Output is [B, 2304, 1, 1, 1] typically
            feats = feats.mean(dim=[2, 3, 4])  # Pool to [B, 2304]
            projected = self.proj(feats)       # [B, 768]
        return projected

    def prompt_definition(self, question: str, system_message: str = "You are a helpful assistant.", video: list = None):
        prompt = f"USER: <video>\n{system_message}\n{question}\nASSISTANT:"
        return prompt

    def process_video_tensor(self, video_tensor):
        """
        Converts raw video tensor into LLaVA-ready format by extracting SlowFast features.
        """
        video_tensor = video_tensor.to(self.device).float()  # Ensure correct device & dtype
        features = self.extract_features(video_tensor)
        return features

    def clean_answer(self, answer: str):
        answer = answer.split("ASSISTANT:")[-1].strip()
        return answer
