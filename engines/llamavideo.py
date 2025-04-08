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
        self.processor = VideoLlavaProcessor.from_pretrained(base_model_id)
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
    
    def prompt_definition(self, question: str):
        """
        Build the prompt text for a given question.
        Here, we follow the recommended prompt format for Video LLaVA.
        """
        # For example, the prompt is defined with a video token placeholder.
        prompt = f"USER: <video>\n{question}\nASSISTANT:"
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
        
        for question in questions:
            # Create the prompt text using the given question.
            prompt_text = self.prompt_definition(question)
            
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
            match = re.search(r'\b[01]\b', final_answer)
            if match:
                final_answer = match.group()
            elif final_answer.lower().startswith("yes"):
                final_answer = "1"
            elif final_answer.lower().startswith("no"):
                final_answer = "0"
            else:
                print(answer, flush=True)
                final_answer = '2'
        
            responses.append(final_answer)
            full_answers.append(answer)
        
        return responses, full_answers
