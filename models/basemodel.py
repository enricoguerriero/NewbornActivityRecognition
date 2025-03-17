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
    
    def preprocess_videos(self, set_name: str, clip_length: int = 3, frames_per_second: int = 5, overlap: float = 0.5, event_categories: list[str] = []):
        """
        Preprocess videos for a given dataset.
        :param set_name: Name of the dataset (e.g., 'train', 'validation', 'test').
        :param clip_length: Length of video clips in seconds.
        :param frames_per_second: Number of frames per second.
        :param overlap: Overlap between consecutive clips.
        :param event_categories: List of event categories to consider.
        """

        logger = logging.getLogger(f'{self.model_name}_preprocessing')
        logger.info(f"Preprocessing videos for {set_name} set.")
        
        logger.debug(f'Input parameters: {set_name}, {clip_length}, {frames_per_second}, {overlap}, {event_categories}')
        
        video_folder = os.path.join(self.video_folder, set_name)
        annotation_folder = os.path.join(self.annotation_folder, set_name)
        output_folder = os.path.join(self.output_folder, set_name)
        os.makedirs(output_folder, exist_ok=True)
        logger.info(f"Output folder: {output_folder}")
        
        video_files = sorted(os.listdir(video_folder))
        logger.info(f"Found {len(video_files)} video files.")
        logger.info(f"Event categories: {event_categories}")
        for video_file in tqdm(video_files):
            video_path = os.path.join(video_folder, video_file)
            base_name, _ = os.path.splitext(video_file)
            annotation_path = os.path.join(annotation_folder, base_name + ".txt")
            if not os.path.exists(annotation_path):
                logger.error(f"Annotation file for {video_file} not found, skipping.")
                continue
            logger.debug(f"Preprocessing video: {video_file}")
            self._preprocess_video(video_path, annotation_path, output_folder, 
                                   clip_length=clip_length, frames_per_second=frames_per_second,
                                   overlap=overlap, event_categories=event_categories, logger=logger)
            
    def _preprocess_video(self, video_path, annotation_path, output_folder, 
                          clip_length=3, frames_per_second=5, overlap=0.5, event_categories=[],
                          logger=None):
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logger.error(f"Error opening video file: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            logger.error(f"Invalid FPS for {video_path}, using fallback of 30")
            fps = 30

        # Determine sampling parameters.
        frame_interval = int(round(fps / frames_per_second))
        clip_frame_count = int(clip_length * frames_per_second)
        overlap_frames = int(overlap * frames_per_second)
        if overlap_frames >= clip_frame_count:
            logger.error("Overlap must be less than clip length. Setting overlap to 0.")
            overlap_frames = 0
        hop_frames = clip_frame_count - overlap_frames

        clip_index = 0
        # current_clip stores tuples: (processed_frame, absolute_frame_index)
        current_clip = []
        frame_idx = 0

        logger.debug(f"Sampling parameters computed, starting frame extraction.")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_idx % frame_interval == 0:
                # Process the frame.
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_proc = self.process_frame(frame_rgb)
                current_clip.append((frame_proc, frame_idx))
                if len(current_clip) == clip_frame_count:
                    # Get list of absolute frame indices for this clip.
                    frame_indices = [item[1] for item in current_clip]
                    clip_start_time = (frame_indices[0] / fps) * 1000  # in milliseconds
                    # Generate per-frame binary labels and then average to get clip-level label.
                    labels = self._generate_labels(annotation_path, frame_indices, fps, event_categories, logger)
                    clip_labels = torch.mean(labels.float(), dim=0)
                    # Stack frames into a tensor of shape [num_frames, C, H, W].
                    frames_tensor = torch.stack([item[0] for item in current_clip])
                    clip_data = {
                        'frames': frames_tensor,
                        'labels': clip_labels,
                        'video_path': video_path,
                        'clip_index': clip_index,
                        'clip_start_time': clip_start_time,
                        'clip_length': clip_length,
                        'sampling_rate': frames_per_second
                    }
                    logger.debug(f"tensor dimensions: {frames_tensor.shape}")
                    output_filename = os.path.splitext(os.path.basename(video_path))[0] + f"_clip_{clip_index}.pt"
                    output_filepath = os.path.join(output_folder, output_filename)
                    torch.save(clip_data, output_filepath)
                    logger.debug(f"Saved clip: {output_filepath}")
                    clip_index += 1
                    # Slide the window: keep the last 'overlap_frames' frames.
                    current_clip = current_clip[hop_frames:]
            frame_idx += 1
        cap.release()

    
    def define_transformation(self, target_size: tuple[int, int]):
        """
        Define a transformation to apply to the input frames.
        :param target_size: Target size for the frames.
        :return: A transformation object.
        """
        return transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(target_size),
            # transforms.ToTensor()
        ])
        
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

    def process_frame(self, frame_rgb):
        """
        Process a single frame (RGB) before passing to the model.
        """
        frame_proc = self.transform(frame_rgb)
        if self.image_processor is not None:
            frame_proc = self.image_processor(frame_proc, return_tensors="pt")["pixel_values"].squeeze()
        else:
            frame_proc = torch.tensor(np.array(frame_proc)).permute(2, 0, 1).float() / 255.0
        return frame_proc
    
    
    def _generate_labels(self, annotation_path, frame_indices, fps, event_categories, logger = None):
        """
        Generate per-frame labels for a clip given the absolute frame indices.
        Returns a tensor of shape [num_frames, num_events] with binary values.
        """
        num_frames = len(frame_indices)
        num_categories = len(event_categories)
        labels = np.zeros((num_frames, num_categories), dtype=np.int64)
        # Compute the absolute time in milliseconds for each frame.
        frame_times = np.array([(idx / fps) * 1000 for idx in frame_indices])
        try:
            with open(annotation_path, 'r') as f:
                lines = f.readlines()
        except Exception as e:
            logger.error(f"Error reading annotation file {annotation_path}: {e}")
            return torch.tensor(labels, dtype=torch.int32)

        for line in lines:
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            if len(tokens) < 4:
                continue
            event_name = " ".join(tokens[:-3]).strip()
            try:
                event_start = float(tokens[-3])
                event_end = float(tokens[-2])
            except ValueError:
                continue
            if event_name in event_categories:
                event_idx = event_categories.index(event_name)
                indices = np.where((frame_times >= event_start) & (frame_times < event_end))[0]
                labels[indices, event_idx] = 1
        return torch.tensor(labels, dtype=torch.int32)
    
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