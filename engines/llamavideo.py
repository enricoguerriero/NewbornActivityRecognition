import torch
import re
from transformers import VideoLlavaForConditionalGeneration, VideoLlavaProcessor
from engines.prompt_engine import PromptEngine 

class VideoLLamaEngine(PromptEngine):
    """
    A wrapper for the Video LLaVA model that encapsulates loading the processor and model,
    as well as generating answers from prompts, with an interface similar to the SmolVLMEngine.
    """
    def __init__(self, checkpoint_path: str = None, base_model_id: str = "LanguageBind/Video-LLaVA-7B-hf", device=None):
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load the processor and model
        self.prompt_processor = VideoLlavaProcessor.from_pretrained(base_model_id)
        if checkpoint_path:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                checkpoint_path,
                torch_dtype=torch.float16
            ).to(self.device)
        else:
            self.model = VideoLlavaForConditionalGeneration.from_pretrained(
                base_model_id,
                torch_dtype=torch.float16
            ).to(self.device)
        
        self.model.eval()
        self.name = "llama_video"
    
    def prompt_definition(self, question: str, image_tokens: list):
        """
        Build the prompt text for a given question.
        Here, we follow the recommended prompt format for Video LLaVA.
        """
        prompt = [
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
        
        return prompt
    
    def answer_questions(self, video_list: list, questions: list, seed: int = 42, temperature: float = 0.1):
        """
        Given a list of videos (as numpy arrays) and a list of questions, generate answers.
        
        :param video_list: List of video numpy arrays.
        :param questions: List of question strings.
        :param seed: Random seed for reproducibility.
        :param temperature: Sampling temperature.
        :return: Tuple (responses, full_answers) where:
                 - responses is a list of post-processed final answers (extracted binary response: '0' or '1'),
                 - full_answers is a list of the complete generated texts.
        """
        torch.manual_seed(seed)
        
        responses = []
        full_answers = []
        
        image_tokens = [{"type": "image"} for _ in range(len(video_list))]
        
        for question in questions:
            # Create the prompt text using the given question.
            prompt_text = self.prompt_definition(question, image_tokens)
            
            # Process inputs (text and videos).
            inputs = self.processor(text=prompt_text, videos=video_list, return_tensors="pt").to(self.device)
            
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
            # Using batch_decode here so that it returns a list; we extract the first answer.
            answer = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
            
            # Post-process: Remove extra tokens and extract the binary answer (0 or 1).
            final_answer = answer.split("ASSISTANT:")[-1].strip()

            if final_answer.lower().startswith("yes"):
                final_answer = "1"
            elif final_answer.lower().startswith("no"):
                final_answer = "0"
            else:
                print(answer, flush=True)
                final_answer = '2'
        
            responses.append(final_answer)
            full_answers.append(answer)
        
        return responses, full_answers

    def describe_the_scene(self, video_list: list, seed: int = 42, temperature: float = 0.1):
        """
        Given a list of videos (as numpy arrays), generate a detailed description of the scene.
        
        The prompt asks the model to provide a comprehensive description including context,
        observable details, and any events or motions present in the video.
        
        :param video_list: List of video numpy arrays.
        :param seed: Random seed for reproducibility.
        :param temperature: Sampling temperature.
        :return: A string with the generated scene description.
        """
        torch.manual_seed(seed)
        
        # Define a prompt for scene description.
        prompt_text = ("USER: <video>\n"
                       "Please describe the scene in detail. Include the context, any actions or events, "
                       "and all observable details from the video.\n"
                       "ASSISTANT:")
        
        # Process inputs (text and videos)
        inputs = self.processor(text=prompt_text, videos=video_list, return_tensors="pt").to(self.device)
        
        # Generate the scene description using the model.
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=100,
            num_beams=5,
            temperature=temperature,
            do_sample=True,
            use_cache=True
        )
        
        # Decode the generated output.
        description = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        # Extract the description after the "ASSISTANT:" token.
        description = description.split("ASSISTANT:")[-1].strip()
        
        return description
