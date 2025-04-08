from transformers import AutoModelForCausalLM
from janus.models import MultiModalityCausalLM, VLChatProcessor
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
        self.vl_chat_processor = VLChatProcessor.from_pretrained(model_id)
        self.tokenizer = self.vl_chat_processor.tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            trust_remote_code=True,
            torch_dtype=torch.float16,
            device_map="auto"
        ).to(self.device)
        
        # Set the model to evaluation mode
        self.model.eval()
        
        self.name = "janus_pro"

    def prompt_definition(self, question: str, image: Image):
        """
        Define the prompt structure for the model.
        """
        conversation = [
            {
                "role": "<|User|>",
                "content": f"<image_placeholder>\n {question}",
                "images": [image],
            },
            {"role": "<|Assistant|>", "content": ""},
        ]

        return conversation

    def answer_questions(self, image_list: list, questions: list[str], seed: int = 42, temperature: float = 0.1):
        """
        Given a PIL image and a question, generate an answer.
        """
        torch.manual_seed(seed)
        
        # take the first image from the list
        image = image_list[len(image_list) // 2]
        responses = []
        full_answers = []
        
        for question in questions:
        
            # Define the prompt for the current question
            prompt_text = self.prompt_definition(question, image)
            print(prompt_text, flush=True)
            
            # Process inputs (both text and image)
            inputs = self.vl_chat_processor(
                conversation=prompt_text,
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

            responses.append(final_answer)
            full_answers.append(answer)

        return final_answer, answer
