import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm

class VideoActivityRecognitionModel(nn.Module, ABC):
    """
    Generic base class for video activity recognition models.
    Provides generic training and testing methods.
    """

    def __init__(self, device='cuda'):
        """
        Initializes the base model.
        
        Args:
            device (str): Device to run on (e.g., 'cuda' or 'cpu').
            wandb_project (str): Name of the wandb project. If provided, wandb is initialized.
            wandb_config (dict): Configuration dictionary for wandb.
        """
        super(VideoActivityRecognitionModel, self).__init__()
        # Set device (fallback to CPU if CUDA is not available)
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

    @abstractmethod
    def forward(self, x):
        """
        Forward pass.
        Must be implemented in subclasses.
        """
        pass

    def predict(self, pixel_values):
        """
        Predicts multi-label outputs using a probability threshold.
        
        Args:
            pixel_values (torch.Tensor): Input video frames.
        
        Returns:
            predictions (torch.Tensor): Binary tensor (0/1) of shape (batch_size, num_event_classes)
                                        where 1 indicates the event is predicted.
        """
        logits = self.forward(pixel_values)
        probs = torch.sigmoid(logits)
        predictions = (probs > self.threshold).int()
        return predictions


    def train_epoch(self, dataloader, optimizer, criterion, epoch, verbose=True):
        """
        Trains the model for one epoch.
        
        Args:
            dataloader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer.
            criterion (torch.nn.Module): Loss function (recommended: BCEWithLogitsLoss for multi-label).
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

        for frames, labels in progress:
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

    def test(self, dataloader, criterion, logger, wandb):
        """
        Evaluates the model on the test set.
        
        Args:
            dataloader (DataLoader): DataLoader for test data.
            criterion (torch.nn.Module): Loss function.
        
        Returns:
            avg_loss (float): Average test loss.
            accuracy (float): Average per-label accuracy percentage.
        """
        if logger and wandb:
            logger.debug("Evaluating model on test set...")
        self.eval()  
        running_loss = 0.0
        total_labels = 0
        total_correct = 0

        with torch.no_grad():
            for frames, labels in dataloader:
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
        logger.debug(f"Test Loss = {avg_loss:.4f} | Test Acc = {accuracy:.2f}%")
        wandb.log({"test_loss": avg_loss, "test_accuracy": accuracy})
        return avg_loss, accuracy

    def train_model(self, train_loader, optimizer, criterion, num_epochs, learning_rate, momentum, val_loader=None, logger=None, wandb=None):
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
        history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

        epoch_iter = tqdm(range(1, num_epochs + 1), desc="Training Epochs")
        for epoch in epoch_iter:
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer, criterion, epoch)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            log_message = f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Train Acc = {train_accuracy:.2f}%"

            if val_loader is not None:
                val_loss, val_accuracy = self.test(val_loader, criterion)
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
        
        return history

    def save_model(self, path):
        """
        Saves the model to a file.
        
        Args:
            path (str): Path to save the model.
        """
        save_path = "baseline/saved" + path + ".pt"
        torch.save(self.state_dict(), save_path)
        
    def load_model(self, path):
        """
        Loads the model from a file.
        
        Args:
            path (str): Path to load the model from.
        """
        load_path = "baseline/saved" + path + ".pt"
        self.load_state_dict(torch.load(load_path))
        
    def save_checkpoint(self, path, optimizer, epoch):
        """
        Saves a model checkpoint to a file.
        
        Args:
            path (str): Path to save the checkpoint.
            optimizer (torch.optim.Optimizer): Optimizer.
            epoch (int): Current epoch.
        """
        save_path = "baseline/saved/checkpoints" + path + ".pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)