from models.LLMs.video_understanding_model import VideoUnderstandingModel

import torch
from transformers import AutoProcessor, Idefics3ForConditionalGeneration
from torchvision import transforms

class SmolVLMModel(VideoUnderstandingModel):
    """
    A subclass of ImageUnderstandingModel that implements the SmolVLM model
    for answering questions about images.
    """
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "HuggingFaceTB/SmolVLM-Instruct", device=None):
        super().__init__(device)
        
        # Load processor and model
        self.prompt_processor = AutoProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )
        else:
            self.model = Idefics3ForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.bfloat16,
                device_map=self.device
            )
        
        self.prompt_processor.image_processor.size = (384, 384)
        self.prompt_processor.image_processor.do_resize = False
        self.prompt_processor.image_processor.do_image_splitting = False
        self.model_name = "smolvlm"
    
    def answer_questions(self, video_tensor, questions: list, seed: int = 42,
                        top_p: float = 0.95, temperature: float = 0.1) -> str:
        """
        Given an image and a question, generate an answer using SmolVLM.
        """
        torch.manual_seed(seed)
        
        to_pil = transforms.ToPILImage()
        image_list = [to_pil(frame) for frame in video_tensor]
        
        image_tokens = [{"type": "image"} for _ in range(len(image_list))]
        
        response = []
        
        for question in questions:
            
            prompt = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Answer using just 0 or 1 following the instruction."},
                        *image_tokens,
                        {"type": "text", "text": question}
                    ]
                }
            ]
            
            inputs = self.prompt_processor(
                text=self.prompt_processor.apply_chat_template(prompt, add_generation_prompt=True),
                images=image_list,
                return_tensors="pt"
            )
            
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                num_beams=5,
                temperature=0.7,
                do_sample=True,
                use_cache=True
            )
            
            response.append(self.prompt_processor.decode(outputs[0], skip_special_tokens=True))
            
        return response