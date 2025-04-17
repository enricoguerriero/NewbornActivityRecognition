from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn
from models.basemodel import BaseVideoModel  
from tqdm import tqdm
import logging
import torch
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os
from torch.optim.lr_scheduler import StepLR

# Import useful metrics from scikit-learn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class TimesformerModel(BaseVideoModel):
    """
    Timesformer-based model for video activity recognition.
    This subclass uses a pretrained Timesformer backbone and adds a custom classification head.
    Supports multi-label classification where multiple events may occur in the same clip.
    """

    def __init__(self, num_event_classes=4, hidden_size=768, threshold=0.5, device='cuda'):
        """
        Initializes the Timesformer model.
        
        Args:
            num_event_classes (int): Number of event classes.
            hidden_size (int): Hidden size of the transformer.
            threshold (float): Probability threshold for predicting an event.
            device (str): Device to run on.
        """
        super(TimesformerModel, self).__init__(device=device, model_name="timesformer")
        self.device = device  
        self.threshold = threshold  # threshold for multi-label classification

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-ssv2")
        # Load a pretrained Timesformer for video classification.
        self.timesformer = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-ssv2"
        )
        # Remove the original classification head.
        self.timesformer.classifier = nn.Identity()
        
        for param in self.timesformer.parameters():
            param.requires_grad = False
        
        # Custom classification head: outputs logits for each event.
        self.event_classifier = nn.Linear(hidden_size, num_event_classes)
        self.to(self.device)
        self.model_name = "timesformer"
        
        
    def forward(self, pixel_values):
        """
        Forward pass for the Timesformer model.
        
        Args:
            pixel_values (torch.Tensor): Tensor of shape (batch_size, num_frames, 3, 224, 224)
        
        Returns:
            logits (torch.Tensor): Raw logits for each event (before applying sigmoid).
        """
        outputs = self.timesformer(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # shape: [B, seq_len, hidden_size]
        pooled = hidden_states.mean(dim=1)         # mean pooling across sequence
        logits = self.event_classifier(pooled)
        return logits

    def compute_metrics(self, all_predictions, all_labels, all_probs):
        """
        Computes overall and per-event metrics for multi-label classification.
        
        Args:
            all_predictions (np.array): Binary predictions (thresholded output), shape (N, num_events).
            all_labels (np.array): Ground truth binary labels, shape (N, num_events).
            all_probs (np.array): Probabilities computed from the logits, shape (N, num_events).
            
        Returns:
            metrics (dict): Dictionary with overall precision, recall, F1 scores (micro and macro),
                            ROC AUC and average precision scores, plus a "per_event" key containing
                            the computed metrics for each event.
        """
        # Ensure that the binary predictions and labels are integer types.
        all_labels = all_labels.astype(int)
        all_predictions = all_predictions.astype(int)
        metrics = {}

        # Compute per-event metrics
        num_events = all_labels.shape[1]  # for example, 4 events/questions
        per_event_metrics = {}

        for i in range(num_events):
            # Extract values for event i
            y_true = all_labels[:, i].astype(int)
            y_pred = all_predictions[:, i].astype(int)
            y_probs = all_probs[:, i]  # as floats for probability scores

            event_dict = {}
            event_dict["accuracy"] = np.mean(y_true == y_pred)  # per-question accuracy
            event_dict["precision"] = precision_score(y_true, y_pred, zero_division=0)
            event_dict["recall"] = recall_score(y_true, y_pred, zero_division=0)
            event_dict["f1_score"] = f1_score(y_true, y_pred, zero_division=0)

            try:
                event_dict["roc_auc"] = roc_auc_score(y_true, y_probs)
            except Exception as e:
                event_dict["roc_auc"] = None

            try:
                event_dict["average_precision"] = average_precision_score(y_true, y_probs)
            except Exception as e:
                event_dict["average_precision"] = None

            per_event_metrics[f"event_{i}"] = event_dict

        metrics["per_event"] = per_event_metrics
        return metrics

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
            extra_metrics (dict): Additional computed metrics.
        """
        self.train()  
        running_loss = 0.0
        total_labels = 0
        total_correct = 0

        # Lists to accumulate predictions, probabilities, and labels for the entire epoch.
        all_labels_list = []
        all_preds_list = []
        all_probs_list = []

        progress = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False) if verbose else dataloader

        for batch in progress:
            
            frames = batch['frames'].to(self.device)
            labels = batch['labels'].to(self.device).float()

            optimizer.zero_grad()
            outputs = self.forward(frames)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            
            # Compute predictions using sigmoid and thresholding
            probs = torch.sigmoid(outputs)
            predictions = (probs > self.threshold).float()
            total_correct += (predictions == labels).sum().item()
            total_labels += labels.numel()
            
            # Accumulate for metrics calculation
            all_labels_list.append(labels.detach().cpu())
            all_preds_list.append(predictions.detach().cpu())
            all_probs_list.append(probs.detach().cpu())

        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * total_correct / total_labels if total_labels > 0 else 0
        
        # Concatenate all predictions and labels from the epoch.
        all_labels_tensor = torch.cat(all_labels_list, dim=0)
        all_preds_tensor = torch.cat(all_preds_list, dim=0)
        all_probs_tensor = torch.cat(all_probs_list, dim=0)
        
        all_labels_np = all_labels_tensor.numpy()
        all_preds_np = all_preds_tensor.numpy()
        all_probs_np = all_probs_tensor.numpy()
        
        extra_metrics = self.compute_metrics(all_preds_np, all_labels_np, all_probs_np)
        return avg_loss, accuracy, extra_metrics

    def test(self, dataloader, criterion, logger=None, wandb=None):
        """
        Evaluates the model on the test set.
        
        Args:
            dataloader (DataLoader): DataLoader for test data.
            criterion (torch.nn.Module): Loss function.
        
        Returns:
            avg_loss (float): Average test loss.
            accuracy (float): Average per-label accuracy percentage.
            extra_metrics (dict): Additional computed metrics.
        """
        if logger is None:
            logger = logging.getLogger(f'{self.model_name}_test')
        
        if logger and wandb:
            logger.debug("Evaluating model on test set...")
        self.eval()  
        running_loss = 0.0
        total_labels = 0
        total_correct = 0
        
        # Lists to accumulate predictions and labels.
        all_labels_list = []
        all_preds_list = []
        all_probs_list = []
        
        with torch.no_grad():
            for batch in dataloader:
                frames = batch['frames'].to(self.device)
                labels = batch['labels'].to(self.device).float()
                outputs = self.forward(frames)
                loss = criterion(outputs, labels)
                running_loss += loss.item()

                probs = torch.sigmoid(outputs)
                predictions = (probs > self.threshold).float()
                total_correct += (predictions == labels).sum().item()
                total_labels += labels.numel()
                
                all_labels_list.append(labels.detach().cpu())
                all_preds_list.append(predictions.detach().cpu())
                all_probs_list.append(probs.detach().cpu())

        avg_loss = running_loss / len(dataloader)
        accuracy = 100 * total_correct / total_labels if total_labels > 0 else 0
        
        all_labels_tensor = torch.cat(all_labels_list, dim=0)
        all_preds_tensor = torch.cat(all_preds_list, dim=0)
        all_probs_tensor = torch.cat(all_probs_list, dim=0)
        
        all_labels_np = all_labels_tensor.numpy()
        all_preds_np = all_preds_tensor.numpy()
        all_probs_np = all_probs_tensor.numpy()
        
        extra_metrics = self.compute_metrics(all_preds_np, all_labels_np, all_probs_np)
        logger.debug(f"Test / Val Loss = {avg_loss:.4f} | Test / Val Acc = {accuracy:.2f}%")
        
        if wandb:
            # Log per-clip predictions and ground truth values.
            wandb.log({
                "test_loss": avg_loss, 
                "test_accuracy": accuracy,
                "raw_probabilities": wandb.Histogram(all_probs_np.flatten()),
            })
            
            # Flatten and log per-question metrics (accuracy, precision, recall, F1)
            for i, metrics_dict in extra_metrics["per_event"].items():
                wandb.log({
                    f"test_{i}_accuracy": metrics_dict["accuracy"],
                    f"test_{i}_precision": metrics_dict["precision"],
                    f"test_{i}_recall": metrics_dict["recall"],
                    f"test_{i}_f1_score": metrics_dict["f1_score"],
                })
                
                y_true = all_labels_np[:, i]
                y_pred = all_preds_np[:, i]
                cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

                # Plot the confusion matrix using seaborn heatmap
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                ax.set_title(f"Confusion Matrix for Event {i}")
                ax.set_xlabel("Predicted Label")
                ax.set_ylabel("True Label")
                wandb.log({f"confusion_matrix_event_{i}": wandb.Image(fig)})
                plt.close(fig)
                
        return avg_loss, accuracy, extra_metrics

    def train_model(self, train_loader, optimizer, criterion, num_epochs, val_loader=None, wandb=None,
                early_stopping_patience=5, early_stopping_delta=0.0):
        """
        Runs the full training cycle over multiple epochs with early stopping.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer.
            criterion (torch.nn.Module): Loss function.
            num_epochs (int): Maximum number of epochs.
            val_loader (DataLoader, optional): DataLoader for validation/test data.
            wandb: Weights & Biases object for logging metrics.
            early_stopping_patience (int): Number of epochs with no improvement after which training will be stopped.
            early_stopping_delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        
        Returns:
            history (dict): Training and validation loss, accuracy, and metric history.
        """
        logger = logging.getLogger(f'{self.model_name}_train')
        history = {
            "train_loss": [],
            "train_accuracy": [],
            "train_metrics": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_metrics": []
        }

        best_val_loss = float('inf')
        epochs_without_improvement = 0
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        epoch_iter = tqdm(range(1, num_epochs + 1), desc="Training Epochs")
        for epoch in epoch_iter:
            train_loss, train_accuracy, train_metrics = self.train_epoch(train_loader, optimizer, criterion, epoch)
            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_accuracy)
            history["train_metrics"].append(train_metrics)
            log_message = (f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Train Acc = {train_accuracy:.2f}% | "
                           f"Metrics: {train_metrics}")

            if val_loader is not None:
                val_loss, val_accuracy, val_metrics = self.test(val_loader, criterion, logger)
                history["val_loss"].append(val_loss)
                history["val_accuracy"].append(val_accuracy)
                history["val_metrics"].append(val_metrics)
                log_message += (f" | Val Loss = {val_loss:.4f} | Val Acc = {val_accuracy:.2f}% | "
                                f"Metrics: {val_metrics}")
                tqdm_message = f"Epoch {epoch}: Train Loss = {train_loss:.4f} | Train Acc = {train_accuracy:.2f}% | " \
                               f"Val Loss = {val_loss:.4f} | Val Acc = {val_accuracy:.2f}%"
                
                if val_loss < best_val_loss - early_stopping_delta:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1
                    logger.info(f"No improvement in validation loss for {epochs_without_improvement} epoch(s).")
                
                # Stop training early if no improvement for a specified patience
                if epochs_without_improvement >= early_stopping_patience:
                    logger.info(f"Early stopping triggered after {epoch} epochs.")
                    break
                
            scheduler.step()
            
            if logger:
                logger.debug(log_message)
            self.save_checkpoint(f"timesformer_{epoch}", optimizer, epoch)
            if wandb:
                # Log all metrics
                wandb.log({
                    "train_loss": train_loss,
                    "train_accuracy": train_accuracy,
                    **{f"train_{k}": v for k, v in train_metrics.items()},
                    "val_loss": val_loss if val_loader is not None else None,
                    "val_accuracy": val_accuracy if val_loader is not None else None,
                    **({f"val_{k}": v for k, v in val_metrics.items()} if val_loader is not None else {}),
                    "epoch": epoch
                })
            epoch_iter.set_postfix_str(tqdm_message)
            
        # Save the final model
        self.save_model("timesformer_final")
        
        return history

    def save_model(self, path):
        """
        Saves the model to a file.
        
        Args:
            path (str): Path to save the model.
        """
        # Ensure proper path concatenation; you may also consider using os.path.join.
        save_path = "models/saved/" + path + ".pt"
        torch.save(self.state_dict(), save_path)
        
    def load_model(self, path):
        """
        Loads the model from a file.
        
        Args:
            path (str): Path to load the model from.
        """
        load_path = os.path.join("models/saved", path + ".pt")
        # set weights only true
        self.load_state_dict(torch.load(load_path, weights_only=True))
        
    def save_checkpoint(self, path, optimizer, epoch):
        """
        Saves a model checkpoint to a file.
        
        Args:
            path (str): Path to save the checkpoint.
            optimizer (torch.optim.Optimizer): Optimizer.
            epoch (int): Current epoch.
        """
        save_path = os.path.join("models/saved", path + "_checkpoint.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
