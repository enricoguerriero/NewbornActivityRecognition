from transformers import AutoModelForCausalLM, AutoConfig
from janus.models import VLChatProcessor
import torch
from PIL import Image
import re

class JanusProEngine:
    """
    A wrapper for the Janus-Pro model that encapsulates loading the processor and model,
    as well as generating answers from prompts.
    """
    def __init__(self, model_id: str = "deepseek-ai/Janus-Pro-7B", device: str = None):
        self.device = torch.device(device if device else ('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load model configuration and adjust language settings.
        config = AutoConfig.from_pretrained(model_id)
        language_config = config.language_config
        language_config._attn_implementation = 'eager'
        
        # Load the model.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            language_config=language_config,
            trust_remote_code=True
        ).to(self.device)
        
        # Load the processor.
        self.processor = VLChatProcessor.from_pretrained(model_id)
        
        # Set the model to evaluation mode.
        self.model.eval()
        
        self.name = "janus_pro"

    def prompt_definition(self, question: str, system_message, video: Image):
        """
        Define the prompt structure for a question.
        """
        context = system_message
        image = video
        conversation = [
            {
                "role": "<|User|>",
                "content": f"{context}<image_placeholder>\n{question}",
                "images": [image],
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            },
        ]
        return conversation

    def process_frames(self, frames):
        """
        Process the frames into a format suitable for the model.
        """
        images = [Image.fromarray(frame.cpu().numpy()) for frame in frames]
        
        image = images[len(images) // 2]
        return image
    
    def clean_answer(self, answer):
        """
        Clean the answer string to extract the relevant information.
        """
        try:
            answer = answer.split("<|Assistant|>:")[-1]
            return answer
        except:
            return answer