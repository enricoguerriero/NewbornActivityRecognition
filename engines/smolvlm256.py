from transformers import AutoProcessor
from transformers import AutoModelForVision2Seq
from engines.smolvlm import SmolVLMEngine
import torch
import re

class SmolVLM256Engine(SmolVLMEngine):
    """
    A wrapper for the smolvlm model that encapsulates loading the processor and model,
    as well as generating answers from prompts.
    """
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "HuggingFaceTB/SmolVLM-256M-Instruct", device=None):
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model.
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.model = AutoModelForVision2Seq.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
            ).to(self.device)
        else:
            self.model = AutoModelForVision2Seq.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                _attn_implementation="flash_attention_2",
            ).to(self.device)
        
        # Set the model to evaluation mode.
        self.model.eval()
        
        self.name = "smolvlm256"
        
    
    