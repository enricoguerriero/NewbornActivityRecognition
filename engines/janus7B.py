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
        
        config = AutoConfig.from_pretrained(model_id)
        language_config = config.language_config
        language_config._attn_implementation = 'eager'
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            language_config=language_config,
            trust_remote_code=True
        ).to(self.device)
        
        self.processor = VLChatProcessor.from_pretrained(model_id)
        
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
                "content": f"<image_placeholder>\n{question}",
                "images": [image],
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            },
        ]

        return conversation

    def answer_questions(self, image_list: list, questions: list[str], seed: int = 42, temperature: float = 0.1):
        """
        Given a PIL image and a question, generate an answer.
        """
        torch.manual_seed(seed)
        top_p = 0.95 # TO TUNE!
        
        # take the first image from the list
        image = image_list[len(image_list) // 2]
        responses = []
        full_answers = []
        
        for question in questions:
        
            # Define the prompt for the current question
            conversation = self.prompt_definition(question, image)
            
            prepare_inputs = self.processor(conversations=conversation, images=[image], force_batchify=True).to(self.device)
            prepare_inputs['pixel_values'] = prepare_inputs['pixel_values'].to(torch.float32)
            
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            
            # Generate answer using the model
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=temperature > 0,
                use_cache=True,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Decode the generated output
            answer = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
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
