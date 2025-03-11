import torch
import numpy as np
import re
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from engines.prompt_engine import PromptEngine 
from PIL import Image

class VideoLlavaEngine(PromptEngine):
    """
    A wrapper for the VideoLlava model that encapsulates loading the processor and model,
    as well as generating answers from video-based prompts.
    
    The answer_questions method accepts a list of PIL images (treated as consecutive video frames)
    and a list of questions, and returns a list of generated answers.
    """
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf", device=None):
        # Set the device: use provided device, or default to cuda if available, else cpu.
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model.
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16,
                device_map="auto"
            ).to(self.device)
        else:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16,
                device_map="auto"
            ).to(self.device)
        
        # Set the model to evaluation mode.
        self.model.eval()
        self.name = "video_llava"
    
    def prompt_definition(self, question: str):
        """
        Constructs the prompt string using the recommended VideoLLaVA format.
        
        The prompt format is:
            "USER: <video>\n{question} ASSISTANT:"
        """
        return f"USER: <video>\n{question} ASSISTANT:"
    
    def answer_questions(self, image_list: list, questions: list, seed: int = 42, temperature: float = 0.1):
        """
        Given a list of PIL images (treated as consecutive video frames) and a list of questions,
        generate answers for each question.
        
        :param image_list: List of PIL.Image instances.
        :param questions: List of question strings.
        :param seed: Random seed for reproducibility.
        :param temperature: Sampling temperature.
        :return: List of generated answers (one per question).
        """
        torch.manual_seed(seed)
        responses = []
        
        # Convert the list of PIL images into a numpy array of shape (num_frames, height, width, 3)
        # which serves as the video input.
        video = np.stack([np.array(img) for img in image_list], axis=0)
        
        for question in questions:
            # Construct the prompt using the video-specific format.
            prompt_text = self.prompt_definition(question)
            
            # Process inputs: both text and video.
            inputs = self.processor(text=prompt_text, videos=video, return_tensors="pt").to(self.device)
            
            # Generate answers using the model.
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=60,
                num_beams=5,
                temperature=temperature,
                do_sample=True,
                use_cache=True
            )
            
            # Decode the generated output.
            decoded_output = self.processor.batch_decode(
                outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )[0]
            
            # Extract the assistant's answer from the output.
            final_answer = decoded_output.split("ASSISTANT:")[-1].strip()
            
            responses.append(final_answer)
        
        return responses
