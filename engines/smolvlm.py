from transformers import AutoProcessor
from transformers import Idefics3ForConditionalGeneration
from engines.prompt_engine import PromptEngine
import torch
import re

class SmolVLMEngine(PromptEngine):
    """
    A wrapper for the smolvlm model that encapsulates loading the processor and model,
    as well as generating answers from prompts.
    """
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "HuggingFaceTB/SmolVLM-Instruct", device=None):
        super().__init__()
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model.
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16
            ).to(self.device)
        else:
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16
            ).to(self.device)
        
        # Adjust image processor settings.
        # self.prompt_processor.image_processor.size = (384, 384)
        # self.prompt_processor.image_processor.do_resize = False
        # self.prompt_processor.image_processor.do_image_splitting = False
        
        # Set the model to evaluation mode.
        self.model.eval()
        
        self.name = "smolvlm"
        
    
    def process_frames(self, frames):
        
        frames = [frame.cpu().numpy() for frame in frames]
        
        return frames
        
    def prompt_definition(self, system_message: str, question: str, video: list = None):
        
        image_tokens = [{"type": "image"} for _ in range(len(video))]
        
        prompt_template = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_message},
                ]
            },
            {
                "role": "user",
                "content": [
                    *image_tokens,
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        prompt = self.processor.apply_chat_template(prompt_template, add_generation_prompt=True)
        
        return prompt
    
    def clean_answer(self, answer):
        return answer.split("Assistant:")[-1].strip()
    

    