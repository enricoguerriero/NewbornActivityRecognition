from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn
from models.basemodel import BaseVideoModel  
from tqdm import tqdm
import logging
import torch

# Import useful metrics from scikit-learn
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

class TimesformerModel(BaseVideoModel):
    """
    Timesformer-based model for video activity recognition.
    This subclass uses a pretrained Timesformer backbone and adds a custom classification head.
    Supports multi-label classification where multiple events may occur in the same clip.
    """

    def __init__(self, num_event_classes=7, hidden_size=768, threshold=0.5, device='cuda'):
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

        self.image_processor = AutoImageProcessor.from_pretrained("facebook/timesformer-base-finetuned-k400")
        # Load a pretrained Timesformer for video classification.
        self.timesformer = TimesformerForVideoClassification.from_pretrained(
            "facebook/timesformer-base-finetuned-k400"
        )
        # Remove the original classification head.
        self.timesformer.classifier = nn.Identity()
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
        Computes additional metrics for multi-label classification.
        
        Args:
            all_predictions (np.array): Binary predictions (thresholded output).
            all_labels (np.array): Ground truth binary labels.
            all_probs (np.array): Probabilities computed from the logits.
            
        Returns:
            metrics (dict): Dictionary with precision, recall, F1 scores (micro and macro),
                            ROC AUC and average precision scores.
        """
        metrics = {}
        metrics["precision_micro"] = precision_score(all_labels, all_predictions, average="micro", zero_division=0)
        metrics["recall_micro"] = recall_score(all_labels, all_predictions, average="micro", zero_division=0)
        metrics["f1_micro"] = f1_score(all_labels, all_predictions, average="micro", zero_division=0)

        metrics["precision_macro"] = precision_score(all_labels, all_predictions, average="macro", zero_division=0)
        metrics["recall_macro"] = recall_score(all_labels, all_predictions, average="macro", zero_division=0)
        metrics["f1_macro"] = f1_score(all_labels, all_predictions, average="macro", zero_division=0)

        # Compute ROC AUC using probabilities; may fail if a label has no positive samples.
        try:
            metrics["roc_auc"] = roc_auc_score(all_labels, all_probs, average="macro")
        except Exception as e:
            metrics["roc_auc"] = None

        try:
            metrics["average_precision_micro"] = average_precision_score(all_labels, all_probs, average="micro")
            metrics["average_precision_macro"] = average_precision_score(all_labels, all_probs, average="macro")
        except Exception as e:
            metrics["average_precision_micro"] = None
            metrics["average_precision_macro"] = None
        
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
            wandb.log({
                "test_loss": avg_loss, 
                "test_accuracy": accuracy, 
                **extra_metrics
            }) 
        return avg_loss, accuracy, extra_metrics

    def train_model(self, train_loader, optimizer, criterion, num_epochs, val_loader=None, wandb=None):
        """
        Runs the full training cycle over multiple epochs.
        
        Args:
            train_loader (DataLoader): DataLoader for training data.
            optimizer (torch.optim.Optimizer): Optimizer.
            criterion (torch.nn.Module): Loss function.
            num_epochs (int): Number of epochs.
            val_loader (DataLoader, optional): DataLoader for validation/test data.
        
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
            if logger:
                logger.debug(log_message)
            self.save_checkpoint("model", optimizer, epoch)
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
            epoch_iter.set_postfix_str(log_message)
            
        # Save the final model
        self.save_model("model")
        
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
        load_path = "models/saved/" + path + ".pt"
        self.load_state_dict(torch.load(load_path))
        
    def save_checkpoint(self, path, optimizer, epoch):
        """
        Saves a model checkpoint to a file.
        
        Args:
            path (str): Path to save the checkpoint.
            optimizer (torch.optim.Optimizer): Optimizer.
            epoch (int): Current epoch.
        """
        save_path = "models/saved/checkpoints/" + path + ".pt"
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, save_path)
