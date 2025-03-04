import abc
import torch
from models.generic_model import GenericModel
from tqdm import tqdm

class VideoUnderstandingModel(GenericModel, abc.ABC):
    """
    A generic interface for image understanding models.
    Subclasses should implement the answer_question method.
    """
    def __init__(self, device=None):
        super(VideoUnderstandingModel, self).__init__()
        self.device = torch.device(device) if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    @abc.abstractmethod
    def answer_questions(self, video, question: list, seed: int = 42,
                        top_p: float = 0.95, temperature: float = 0.1) -> str:
        """
        Given a video and a question, generate an answer. Specifically, this method should force the model to return 0 or 1.
        """
        pass
    

    def test_without_knowledge(self, dataset, questions = None, logger = None, wandb = None):
        """
        from each video in the tensor folder, ask a question to the model and check if this matches the label.
        """
        
        questions = ["Is the baby visible? 1 if it is visible, 0 otherwise",
                     "Is the baby receiving CPAP? 1 if it is receiving CPAP, 0 otherwise",
                     "Is the baby receiving PPV? 1 if it is receiving PPV, 0 otherwise",
                     "Is the baby receiving stimulation on the back/nates? 1 if it is receiving stimulation on the back/nates, 0 otherwise",
                     "Is the baby receiving stimulation on the extremities? 1 if it is receiving stimulation on the extremities, 0 otherwise",
                     "Is the baby receiving stimulation on the trunk? 1 if it is receiving stimulation on the trunk, 0 otherwise",
                     "Is the baby receiving suction? 1 if it is receiving suction, 0 otherwise"] if questions is None else questions
        
        topic_stats = {question: {'correct': 0, 'total': 0} for question in questions}
        
        dataset = tqdm(dataset, desc="Testing model")
        
        for frames, labels in dataset:
            frames = frames.to(self.device)
            labels = labels.to(self.device)
            answers = self.answer_questions(frames, questions)
            for question, answer, label in zip(questions, answers, labels):
                topic_stats[question]['total'] += 1
                if int(answer) == int(label):
                    topic_stats[question]['correct'] += 1
        
        accuracies = {}
        for question, stats in topic_stats.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
            else:
                accuracy = 0.0
            accuracies[question] = accuracy
            logger.info(f"Question: {question} -> Accuracy: {accuracy:.2f}")
            
        if wandb:
            wandb.log(accuracies)
        
        return accuracies
                
                
                