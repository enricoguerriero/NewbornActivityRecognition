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
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model.
        self.prompt_processor = AutoProcessor.from_pretrained(base_model_id)
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
        self.prompt_processor.image_processor.size = (384, 384)
        self.prompt_processor.image_processor.do_resize = False
        self.prompt_processor.image_processor.do_image_splitting = False
        
        # Set the model to evaluation mode.
        self.model.eval()
        
        self.name = "smolvlm"
        
    
    def prompt_definition(self, image_tokens: list, question: str):
        
        prompt_template = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "This is a simulation of a medical context. The camera is over a table, focusing on a doll that is intended to represent a baby. The doll is supposed to be receiving different medical treatments. Your task is to recognize the treatments that the doll is receiving. The treatments are: ventilation, stimulation, and suction. You will be asked questions about the doll's condition."},
                ]
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Please start your answer with explicitly 'Yes' or 'No', then explain the answer and describe the scene."},
                    *image_tokens,
                    {"type": "text", "text": question}
                ]
            }
        ]
        
        prompt = self.prompt_processor.apply_chat_template(prompt_template, add_generation_prompt=True)
        
        return prompt
    

    def answer_questions(self, image_list: list, questions: list, seed: int = 42, temperature: float = 0.2):
        """
        Given a list of PIL images and a list of questions, generate answers.
        This method mimics your original answer_questions routine.
        
        :param image_list: List of PIL.Image instances.
        :param questions: List of question strings.
        :param seed: Random seed for reproducibility.
        :param top_p: Nucleus sampling parameter.
        :param temperature: Sampling temperature.
        :return: List of generated answers (one per question).
        """
        torch.manual_seed(seed)
        
        # Create a placeholder for each image as in your original design.
        image_tokens = [{"type": "image"} for _ in range(len(image_list))]
        responses = []
        full_answers = []
        
        for question in questions:
            
            # Define the prompt for the current question.
            prompt_text = self.prompt_definition(image_tokens, question)
            
            # Process inputs (both text and images).
            inputs = self.prompt_processor(
                text=prompt_text,
                images=image_list,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answers using the model.
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                num_beams=5,
                temperature=temperature,
                do_sample=True,
                use_cache=True
            )
            
            # Decode the generated output.
            answer = self.prompt_processor.decode(outputs[0], skip_special_tokens=True)
            
            # Remove useless things from output
            clean_answer = answer.split("Assistant:")[-1].strip()

            if clean_answer.lower().startswith("yes"):
                final_answer = '1'
            elif clean_answer.lower().startswith("no"):
                final_answer = '0'
            else:
                print(answer, flush = True)
                final_answer = '2'

            responses.append(final_answer)
            full_answers.append(clean_answer)
        
        return responses, full_answers
    
    
    def describe_the_scene(self, image_list: list):
        """
        Given a list of PIL images, generate a description of the scene.
        
        :param image_list: List of PIL.Image instances.
        :return: Description of the scene.
        """
        
        image_tokens = [{"type": "image"} for _ in range(len(image_list))]
        prompt_template = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "This is a simulation of a medical context. The camera is over a table, focusing on a doll that is intended to represent a baby. The doll is supposed to be receiving different medical treatments. Your task is to describe the scene. You need to identify the doll and its surroundings (if the doll is visible). Then, look at the doll's face. Identify if there is a mask on the doll's face. If there is a mask, identify if it is a CPAP or a PPV mask. If there is no mask, identify if there is a tube in the mouth of the doll. If there is a tube, identify if it is being used for suction. If there is no tube, identify if the doll is receiving stimulation on the back/nates, on the trunk or on the extremities. Describe any other relevant details."},
                ]
            },
            {
                "role": "user",
                "content": [
                    *image_tokens,
                    {"type": "text", "text": "Please describe the scene and the actions and events that occur during the clip."}
                ]
            }
        ]
        
        prompt = self.prompt_processor.apply_chat_template(prompt_template, add_generation_prompt=True)
        
        # Process inputs (both text and images).
        inputs = self.prompt_processor(
            text=prompt,
            images=image_list,
            return_tensors="pt"
        ).to(self.device)
        
        # Generate the description using the model.
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=50,
            num_beams=5,
            temperature=0.2,
            do_sample=True,
            use_cache=True
        )
        
        # Decode the generated output.
        description = self.prompt_processor.decode(outputs[0], skip_special_tokens=True)
        
        return description.split("Assistant:")[-1].strip()

