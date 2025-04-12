import torch
import re
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from engines.prompt_engine import PromptEngine 

class VideoLLavaEngine(PromptEngine):
    """
    A wrapper for the Video LLaVA model that encapsulates loading the processor and model,
    as well as generating answers from prompts, with an interface similar to the SmolVLMEngine.
    """
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf", device=None):
        super().__init__()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model
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
        self.name = "llava_video"
    
    def prompt_definition(self, question: str, system_message: str = "You are a helpful assistant.", video: list = None):
        """
        Build the prompt text for a given question.
        Here, we follow the recommended prompt format for Video LLaVA.
        """
        prompt = f"USER: <video>\n{system_message}\n{question}\nASSISTANT:"
        
        return prompt
    
    def process_frames(self, frames):
        # transform from tensor to list of numpy arrays
        frames = [frame.cpu().numpy() for frame in frames]
        return frames
    
    def clean_answer(self, answer: str):
        answer = answer.split("ASSISTANT:")[-1].strip()
        return answer
        
   