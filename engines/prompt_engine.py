import torch

class PromptEngine:
    def __init__(self, output_dim):
        self.output_dim = output_dim
        # Here you could load a pre-trained prompt model or set up your prompt logic.
        self.name = "genericPromptEngine"
        
    def process_frames(self, frames):
        pass
    
    def prompt_definition(self, question, system_message):
        pass 
    
    def answer_question(self, frames, system_message, question, seed = 42, temperature = 0.1):

        torch.manual_seed(seed)
        
        prompt = self.prompt_definition(question, system_message)
        
        frames_input = self.process_frames(frames)
        
        inputs = self.processor(
            text=prompt,
            videos=frames_input,
            return_tensors="pt"
        ).to(self.device)
        
        generate_kwargs = {
            "max_new_tokens": 100,
            "do_sample": False,
            "temperature": temperature,
            "use_cache": True,
        }
        
        outputs = self.model.generate(**inputs, **generate_kwargs)
        
        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        print(responses, flush=True)