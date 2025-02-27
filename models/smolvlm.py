from models.video_understanding_model import VideoUnderstandingModel

import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration

class SmolVLMModel(VideoUnderstandingModel):
    """
    A subclass of ImageUnderstandingModel that implements the SmolVLM model
    for answering questions about images.
    """
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "HuggingFaceTB/SmolVLM-Instruct", device=None):
        super().__init__(device)
        
        # Load processor and model
        self.processor = AutoProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=device
            )
        else:
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map=device
            )
        
        self.processor.image_processor.size = (384, 384)
        self.processor.image_processor.do_resize = False
        self.processor.image_processor.do_image_splitting = False
        self.model_name = "smolvlm"
    
    def answer_question(self, image, question: str, seed: int = 42,
                        top_p: float = 0.95, temperature: float = 0.1) -> str:
        """
        Given an image and a question, generate an answer using SmolVLM.
        """
        torch.manual_seed(seed)
        
        # Format input prompt
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": question}
                ]
            },
        ]
        
        # Apply chat template and prepare inputs
        prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)
        inputs = self.processor(text=prompt, images=[image], return_tensors="pt").to(self.device)
        
        # Generate response
        generated_ids = self.model.generate(
            **inputs, 
            max_new_tokens=500, 
            do_sample=True,
            top_p=top_p,
            temperature=temperature
        )
        
        # Decode output
        generated_texts = self.processor.batch_decode(generated_ids, skip_special_tokens=True)
        
        # Extract only the assistant's response
        response = generated_texts[0].split("\n")[-1].strip()
        return response.replace("Assistant: ", "").strip()
