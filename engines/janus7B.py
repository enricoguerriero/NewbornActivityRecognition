from transformers import AutoTokenizer, AutoModelForCausalLM
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
        
        # Load the processor and model
        self.processor = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        self.name = "janus_pro"

    def prompt_definition(self, question: str):
        """
        Define the prompt structure for the model.
        """
        prompt_template = f"Answer the following question based on the provided image: {question}"
        return prompt_template

    def answer_question(self, image_list: list, question: str, seed: int = 42, temperature: float = 0.1):
        """
        Given a PIL image and a question, generate an answer.
        """
        torch.manual_seed(seed)
        
        # take the first image from the list
        image = Image.fromarray(image_list[len(image_list) // 2])
        
        # Define the prompt for the current question
        # prompt_text = self.prompt_definition(question)
        prompt_text = question
        
        # Process inputs (both text and image)
        inputs = self.processor(
            text=prompt_text,
            images=image,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate answer using the model
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=500,
            temperature=temperature
        )
        
        # Decode the generated output
        answer = self.processor.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the final answer
        final_answer = answer.split("Assistant:")[-1].strip()
        
        if final_answer.lower().startswith("yes"):
            final_answer = "1"
        elif final_answer.lower().startswith("no"):
            final_answer = "0"
        else:
            final_answer = "2"

        return final_answer, answer
