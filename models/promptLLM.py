from models.basemodel import BaseVideoModel
from engines.prompt_engine import PromptEngine
import torch.nn as nn
from torch.utils.data import DataLoader
import torch
import numpy as np
from PIL import Image
import logging
from tqdm import tqdm

class PromptLLMModel(BaseVideoModel):
    def __init__(self, prompt_engine: PromptEngine, num_classes = 7):
        """
        :param prompt_engine: An instance of a prompt engine (e.g., a small LLM handling PIL images).
        :param num_classes: Number of activity classes.
        """
        super(PromptLLMModel, self).__init__(model_name=prompt_engine.name)
        self.prompt_engine = prompt_engine  # This should be callable.
        # A mapping layer that converts prompt output to desired classification logits.
        # self.mapping = nn.Linear(prompt_engine.output_dim, num_classes)
        self.model_name = prompt_engine.name
        self.image_processor = None
    
    def forward(self, x):
        # x is expected to be a list of PIL images (or a batch of lists)
        prompt_output = self.prompt_engine(x)
        return self.mapping(prompt_output)
    
    def modify_last_layer(self, new_layer_config):
        self.mapping = new_layer_config
        return self

    def test_without_knowledge(self, dataloader: DataLoader, questions: list[str] = None, wandb=None):
        """
        Test the model on a dataset without using extra knowledge from the prompt engine.
        For each sample (video clip) from the dataloader, the prompt engine is queried with a list of binary
        questions. The generated (binary) answers are compared against the ground truth.
        
        The dataloader is expected to yield batches where each batch is a dictionary with keys:
            - 'frames': tensor of shape (batch_size, num_frames, channels, height, width)
            - 'labels': tensor of shape (batch_size, num_questions)
        
        :param dataloader: DataLoader yielding video clip samples.
        :param questions: List of questions. If None, a default set is used.
        :param logger: Optional logger for logging outputs.
        :param wandb: Optional wandb instance for logging.
        :return: Dictionary mapping each question to its accuracy.
        """
        logger = logging.getLogger(f'{self.model_name}_test_untrained')
        
        # Default questions if none provided.
        if questions is None:
            questions = [
                "Is the baby visible? 1 if it is visible, 0 otherwise",
                "Is the baby receiving CPAP? 1 if it is receiving CPAP, 0 otherwise",
                "Is the baby receiving PPV? 1 if it is receiving PPV, 0 otherwise",
                "Is the baby receiving stimulation on the back/nates? 1 if it is receiving stimulation on the back/nates, 0 otherwise",
                "Is the baby receiving stimulation on the extremities? 1 if it is receiving stimulation on the extremities, 0 otherwise",
                "Is the baby receiving stimulation on the trunk? 1 if it is receiving stimulation on the trunk, 0 otherwise",
                "Is the baby receiving suction? 1 if it is receiving suction, 0 otherwise"
            ]
        
        # Initialize statistics for each question.
        topic_stats = {question: {'correct': 0, 'total': 0} for question in questions}
        
        # predicted values to return
        predicted_values = {}
        
        self.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Testing", unit="batch"):
                # Expect batch as a dict with keys 'frames' and 'labels'.
                frames_batch = batch['frames']   # Shape: (batch_size, num_frames, channels, height, width)
                labels_batch = batch['labels']     # Shape: (batch_size, num_questions)
                
                batch_size = frames_batch.size(0)
                for i in range(batch_size):
                    # Extract the frames for the current sample.
                    frames_tensor = frames_batch[i]  # Shape: (num_frames, channels, height, width)
                    clip_name = batch[i]['clip_name']
                    
                    # Convert each frame to a PIL image.
                    pil_images = []
                    for frame in frames_tensor:
                        # Convert tensor to numpy array. Permute to (H, W, C) and cast to uint8.
                        np_frame = frame.permute(1, 2, 0).cpu().numpy()
                        # If the frame is not already in uint8, scale/clip accordingly.
                        if np_frame.dtype != np.uint8:
                            np_frame = (255 * np.clip(np_frame, 0, 1)).astype(np.uint8)
                        pil_images.append(Image.fromarray(np_frame))
                    
                    # Convert ground truth labels to strings for comparison.
                    gt_tensor = labels_batch[i]  
                    gt_tensor = (gt_tensor > 0.5).long()  # Convert to binary labels.
                    gt_list = [str(int(val.item())) for val in gt_tensor]
                    
                    # Obtain predictions from the prompt engine.
                    predictions, full_answer = self.prompt_engine.answer_questions(pil_images, questions)
                    logger.debug(f"Predictions: {predictions}, Ground Truth: {gt_list}")
                    predicted_values[clip_name] = full_answer
                    
                    # Update statistics for each question.
                    for idx, question in enumerate(questions):
                        topic_stats[question]['total'] += 1
                        pred_str = predictions[idx].strip()
                        true_str = gt_list[idx].strip()
                        if pred_str == true_str:
                            topic_stats[question]['correct'] += 1
                    
                    # log details.
                    if logger is not None:
                        logger.debug(f"Predictions: {predictions}, Ground Truth: {gt_list}")
                    if wandb is not None:
                        for idx, question in enumerate(questions):
                            wandb.log({f"{question}_prediction": float(predictions[idx]), f"{question}_gt": float(gt_list[idx])})
        
        # Compute accuracy per question.
        accuracies = {}
        for question, stats in topic_stats.items():
            total = stats['total']
            correct = stats['correct']
            accuracies[question] = correct / total if total > 0 else 0.0
            if logger is not None:
                logger.info(f"Question: {question}, Accuracy: {accuracies[question]*100:.2f}%")
            if wandb is not None:
                wandb.log({f"{question}_accuracy": accuracies[question]})
        
        return accuracies, predicted_values
