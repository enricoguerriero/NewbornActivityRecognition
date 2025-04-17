import torch

class PromptEngine:
    def __init__(self):
        self.name = "genericPromptEngine"
        
    def process_frames(self, frames):
        pass
    
    def prompt_definition(self, question, system_message, video = None):
        pass 
    
    def clean_answer(self, answer):
        pass
    
    def get_predictions_single_question(self, answer):
        """
        Answer is a string with that structure:
        '[0, 1, 0, 1]'
        return the vector of predictions
        """
        try:
            answer = answer.replace("[", "").replace("]", "").replace(" ", "")
            predictions = list(map(int, answer.split(",")))
            if len(predictions) != 4:
                predictions = [2, 2, 2, 2]
        except:
            predictions = [2, 2, 2, 2]
        return predictions
    
    def get_prediction(self, answer):
        """
        Answer is a string starting with 'Yes' or 'No'
        return 1 if 'Yes' elif 'No' return 0 else 2
        """
        try:
            if answer.startswith("Yes"):
                return 1
            elif answer.startswith("No"):
                return 0
            else:
                return 2
        except:
            return 2
        
    def process_input(self, prompt, video):
        pass
        
    
    def answer_question(self, frames, system_message, question, seed = 42, temperature = 0.1):

        torch.manual_seed(seed)
        
        video = self.process_frames(frames)
        
        prompt = self.prompt_definition(question, system_message, video)        
            
        inputs = self.process_input(prompt, video)
                    
        generate_kwargs = {
            "max_new_tokens": 20,
            "do_sample": True,
            "temperature": temperature,
            "use_cache": True,
        }
        
        outputs = self.model.generate(**inputs, **generate_kwargs)
        
        responses = self.processor.batch_decode(outputs, skip_special_tokens=True)[0]
        
        answer = self.clean_answer(responses)
        
        predictions = self.get_prediction(answer)
        
        return predictions, answer