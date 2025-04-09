import torch.nn as nn
import logging
import os
from tqdm import tqdm
from torchvision.transforms import transforms
import cv2
import numpy as np
import torch

class BaseVideoModel(nn.Module):
    """
    An abstract base class for video models.
    Provides a common interface for training, inference, and last-layer modifications.
    """
    def __init__(self, device = "cuda", model_name: str = "baseModel"):
        super(BaseVideoModel, self).__init__()
        self.model_name = model_name
        self.video_folder = "data/videos"
        self.annotation_folder = "data/annotations"
        self.output_folder = "data/processed/" + self.model_name
        self.transform = None
        self.image_processor = None
        self.device = torch.device(device)
        self.to(self.device)
        self.threshold = 0.5  # threshold for multi-label classification
    
    def forward(self, x):
        raise NotImplementedError("Subclasses must implement the forward method.")

    def modify_last_layer(self, new_layer_config):
        """
        Modify the last layer(s) of the model.
        :param new_layer_config: A new layer or sequential block to replace the final layer(s).
        """
        raise NotImplementedError("Subclasses must implement modify_last_layer().")
        
    def define_optimizer(self, optimizer_name, learning_rate, momentum):
        """
        Defines the optimizer for the model.
        By now you can choose between Adam and SGD.
        """
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate, momentum=momentum)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not available")
        return optimizer
    
    def define_criterion(self, criterion_name):
        """
        Defines the criterion for the model.
        By now you can choose between BCE and CrossEntropy.
        """
        if criterion_name == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif criterion_name == "crossentropy":
            criterion = torch.nn.CrossEntropyLoss()
        else:
            raise ValueError(f"Criterion {criterion_name} not available")
        return criterion

    
    
    def train_epoch(self, dataloader, optimizer, criterion, epoch, verbose=True):
        """
        Trains the model for one epoch.
        
        Args:
            dataloader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer.
            criterion (torch.nn.Module): Loss function.
            epoch (int): Current epoch.
            verbose (bool): If True, display progress bar.
        
        Returns:
            avg_loss (float): Average training loss for the epoch.
            accuracy (float): Average per-label accuracy percentage.
        """
        self.train()  
        running_loss = 0.0
        total_labels = 0
        total_correct = 0

        progress = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False) if verbose else dataloader

        for batch in progress:
            
            frames = batch['frames']
            labels = batch['labels']
            
            frames = frames.to(self.device)
            # For multi-label tasks, labels should be a float tensor of shape (batch_size, num_event_classes)
            labels = labels.to(self.device).float()

            optimizer.zero_grad()
            outputs = self.forward(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step() # by now it instead of explicit lr and momentum

            running_loss += loss.item()
            
            # Compute predictions using sigmoid and thresholding
            probs = torch.sigmoid(outputs)
            predictions = (probs > self.threshold).float()
            total_correct += (predictions == labels).sum().item()
            total_labels += labels.numel()

        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * total_correct / total_labels if total_labels > 0 else 0
        return avg_loss, accuracy

    def test(self, dataloader, criterion, logger = None, wandb = None):
        """
        Evaluates the model on the test set.
        
        Args:
            dataloader (DataLoader): DataLoader for test data.
            criterion (torch.nn.Module): Loss function.
        
        Returns:
            avg_loss (float): Average test loss.
            accuracy (float): Average per-label accuracy percentage.
        """
        if logger is None:
            logger = logging.getLogger(f'{self.model_name}_test')
        
        if logger and wandb:
            logger.debug("Evaluating model on test set...")
        self.eval()  
        running_loss = 0.0
        total_labels = 0
        total_correct = 0

        with torch.no_grad():
            for batch in dataloader:
                frames = batch['frames']
                labels = batch['labels']
                
                frames = frames.to(self.device)
                labels = labels.to(self.device).float()
                outputs = self.forward(frames)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predictions = (probs > self.threshold).float()
                total_correct += (predictions == labels).sum().item()
                total_labels += labels.numel()

        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * total_correct / total_labels if total_labels > 0 else 0
        logger.debug(f"Test / Val Loss = {avg_loss:.4f} | Test / Val Acc = {accuracy:.2f}%")
        if wandb:
            wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy}) 
        return avg_loss, accuracy

    def train_model(self, train_loader, optimizer, criterion, num_epochs, val_loader=None, wandb=None):
        """
        Runs the full training cycle over multiple epochs.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer.
            criterion (torch.nn.Module): Loss function.
            num_epochs (int): Number of epochs.
            test_loader (DataLoader, optional): DataLoader for test data.
        
        Returns:
            history (dict): Training and validation loss and accuracy history.
        """
        logger = logging.getLogger(f'{self.model_name}_train')
        history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

        epoch_iter = tqdm(range(1, num_epochs + 1), desc="Training Epochs")
        for epoch in epoch_iter:
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer, criterion, epoch)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            log_message = f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Train Acc = {train_accuracy:.2f}%"

            if val_loader is not None:
                val_loss, val_accuracy = self.test(val_loader, criterion, logger)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)
                log_message += f" | Val Loss = {val_loss:.4f} | Val Acc = {val_accuracy:.2f}%"
            if logger:
                logger.debug(log_message)
            self.save_checkpoint("model", optimizer, epoch)
            if wandb:
                wandb.log({
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy, 
                    "val_loss": val_loss if val_loader is not None else None, 
                    "val_accuracy": val_accuracy if val_loader is not None else None,
                    "epoch": epoch
                })
            epoch_iter.set_postfix_str(log_message)
            
        # save model
        self.save_model("model")
        
        return history
    
    def save_model(self, path):
        """
        Saves the model to a file.
        
        Args:
            path (str): Path to save the model.
        """
        save_path = "models/saved" + path + ".pt"
        torch.save(self.state_dict(), save_path)
        
    def load_model(self, path):
        """
        Loads the model from a file.
        
        Args:
            path (str): Path to load the model from.
        """
        load_path = "models/saved" + path + ".pt"
        self.load_state_dict(torch.load(load_path))
        
    def save_checkpoint(self, path, optimizer, epoch):
        """
        Saves a model checkpoint to a file.
        
        Args:
            path (str): Path to save the checkpoint.
            optimizer (torch.optim.Optimizer): Optimizer.
            epoch (int): Current epoch.
        """
        save_path = "models/saved/checkpoints" + path + ".pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)