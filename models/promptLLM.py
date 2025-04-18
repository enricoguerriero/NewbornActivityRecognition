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

    def test_without_knowledge(self, dataloader: DataLoader, questions: list[str] = None, system_message: str = None, wandb=None):
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

        if questions is None:
            questions = [
                "Is the baby visible?",
                "Is the baby receiving ventilation?",
                "Is the baby receiving stimulation?",
                "Is the baby receiving suction?"
            ]
        if system_message is None:
            system_message = "You are a newborn activity recognition model. Answer the questions based on the video frames."
        
        # Initialize statistics for each question.
        topic_stats = {
            question: {'correct': 0, 'total': 0, 'TP': 0, 'TN': 0, 'FP': 0, 'FN': 0} 
            for question in questions
        }
        
        # predicted values to return
        predicted_values = {}
        
        self.eval()
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Testing", unit="batch"):
                # Expect batch as a dict with keys 'frames' and 'labels'.
                frames_batch = batch['frames']   # Shape: (batch_size, num_frames, channels, height, width)
                labels_batch = batch['labels']     # Shape: (batch_size, num_questions)
                clip_names = batch['clip_name']

                batch_size = frames_batch.size(0)
                for i in range(batch_size):
                    # Extract the frames for the current sample.
                    frames_tensor = frames_batch[i]  # Shape: (num_frames, channels, height, width)
                    clip_name = clip_names[i]
                    
                    # Convert ground truth labels to strings for comparison.
                    gt_tensor = labels_batch[i]  
                    gt_tensor = (gt_tensor > 0.5).long()  # Convert to binary labels.
                    gt_list = [int(val.item()) for val in gt_tensor]
                    
                    # Obtain predictions from the prompt engine.
                    predictions = []
                    for question in questions:
                        prediction, full_answer = self.prompt_engine.answer_question(frames_tensor, system_message, question)
                        predictions.append(prediction)
                        logger.debug(f"Full Answer: {full_answer}")
                    logger.debug(f"Predictions: {predictions}, Ground Truth: {gt_list}")
                    predicted_values[clip_name] = predictions
                    
                    try:
                        # Update statistics for each question.
                        for idx, question in enumerate(questions):
                            topic_stats[question]['total'] += 1
                            pred_str = predictions[idx]
                            true_str = gt_list[idx]
                                                        
                            # Update based on binary outcomes.
                            if true_str == 1:
                                if pred_str == 1:
                                    topic_stats[question]['TP'] += 1
                                    topic_stats[question]['correct'] += 1
                                else:
                                    topic_stats[question]['FN'] += 1
                            elif true_str == 0:
                                if pred_str == 0:
                                    topic_stats[question]['TN'] += 1
                                    topic_stats[question]['correct'] += 1
                                else:
                                    topic_stats[question]['FP'] += 1
                            if pred_str == 2:
                                # for wrong format prediction
                                topic_stats[question]['WF'] += 1
                    except:
                        logger.debug(f"Error processing predictions: {predictions}, Ground Truth: {gt_list}")
                        
                    # log details.
                    if logger is not None:
                        logger.debug(f"Predictions: {predictions}, Ground Truth: {gt_list}")
                    if wandb is not None:
                        for idx in range(4):
                            wandb.log({f"{idx}_prediction": float(predictions[idx]), f"{idx}_gt": float(gt_list[idx])})
        
        metrics = self.compute_metrics(topic_stats, logger, wandb)
            
        return metrics, predicted_values
    
    def compute_metrics(self, topic_stats, logger, wandb):
        
        metrics = {}
        for question, stats in topic_stats.items():
            total = stats['total']
            TP = stats.get('TP', 0)
            TN = stats.get('TN', 0)
            FP = stats.get('FP', 0)
            FN = stats.get('FN', 0)
            WF = stats.get('WF', 0)
            accuracy = (TP + TN) / total if total > 0 else 0.0
            precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
            recall = TP / (TP + FN) if (TP + FN) > 0 else 0.0
            f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            wf_rate = WF / total if total > 0 else 0.0
            metrics[question] = {
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'wf_rate': wf_rate
            }
            
            if logger is not None:
                logger.info(
                    f"Question: {question}, Accuracy: {accuracy*100:.2f}%, "
                    f"Precision: {precision*100:.2f}%, Recall: {recall*100:.2f}%, F1: {f1*100:.2f}%"
                )
            if wandb is not None:
                wandb.log({
                    f"{question}_accuracy": accuracy,
                    f"{question}_precision": precision,
                    f"{question}_recall": recall,
                    f"{question}_f1": f1,
                    f"{question}_wf_rate": wf_rate
                })
            
        return metrics


    