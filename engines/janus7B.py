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

    def prompt_definition(self, question: str, image: Image):
        """
        Define the prompt structure for a question.
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
        Given a list of questions and a list of images, generate answers.
        
        For convenience, one image is chosen from the list (here the middle one) and used
        for all the questions.
        
        :param image_list: List of PIL.Image instances.
        :param questions: List of questions (as strings).
        :param seed: Random seed for reproducibility.
        :param temperature: Sampling temperature.
        :return: Tuple of two lists: (responses, full_answers).
        """
        torch.manual_seed(seed)
        top_p = 0.95  # Nucleus sampling parameter.
        
        # Select a representative image (middle one in the list).
        image = image_list[len(image_list) // 2]
        responses = []
        full_answers = []
        
        for question in questions:
            # Build the conversation prompt for the question.
            conversation = self.prompt_definition(question, image)
            
            # Prepare inputs combining text and image.
            prepare_inputs = self.processor(conversations=conversation, images=[image], force_batchify=True).to(self.device)
            # Ensure the pixel values are in float32.
            prepare_inputs['pixel_values'] = prepare_inputs['pixel_values'].to(torch.float32)
            
            # Prepare embeddings.
            inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
            
            # Generate the answer.
            outputs = self.model.language_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=self.processor.tokenizer.eos_token_id,
                bos_token_id=self.processor.tokenizer.bos_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                max_new_tokens=512,
                do_sample=True,
                use_cache=True,
                temperature=temperature,
                top_p=top_p,
            )
            
            # Decode the generated response.
            answer = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
            
            # Extract the final answer after the assistant's prompt.
            final_answer = answer.split("Assistant:")[-1].strip()
            
            # Convert the answer to a binary code based on the first word.
            if final_answer.lower().startswith("yes"):
                final_answer = "1"
            elif final_answer.lower().startswith("no"):
                final_answer = "0"
            else:
                final_answer = "2"
            
            responses.append(final_answer)
            full_answers.append(answer)
        
        return responses, full_answers

    def describe_the_scene(self, image_list: list, seed: int = 42, temperature: float = 0.1):
        """
        Given a list of PIL images, generate a description of the scene.
        
        The prompt asks the model to detail the observable elements, context,
        objects, and any activities taking place in the scene.
        
        :param image_list: List of PIL.Image instances.
        :param seed: Random seed for reproducibility.
        :param temperature: Sampling temperature.
        :return: A string with the scene description.
        """
        torch.manual_seed(seed)
        # Use one image from the list (middle image) as the representative.
        image = image_list[len(image_list) // 2]
        
        # Build the conversation for scene description.
        conversation = [
            {
                "role": "<|User|>",
                "content": ("Please describe the scene and all observable details, including context, "
                            "objects, and any activities taking place. Provide a detailed description of the environment."),
                "images": [image],
            },
            {
                "role": "<|Assistant|>",
                "content": ""
            },
        ]
        
        # Process inputs (both text and image).
        prepare_inputs = self.processor(conversations=conversation, images=[image], force_batchify=True).to(self.device)
        prepare_inputs['pixel_values'] = prepare_inputs['pixel_values'].to(torch.float32)
        
        # Prepare inputs for generation.
        inputs_embeds = self.model.prepare_inputs_embeds(**prepare_inputs)
        
        # Generate the description using the language model.
        outputs = self.model.language_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=self.processor.tokenizer.eos_token_id,
            bos_token_id=self.processor.tokenizer.bos_token_id,
            eos_token_id=self.processor.tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=True,
            use_cache=True,
            temperature=temperature,
            top_p=0.95,
        )
        
        # Decode the generated output.
        description = self.processor.tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
        description = description.split("Assistant:")[-1].strip()
        
        return description
