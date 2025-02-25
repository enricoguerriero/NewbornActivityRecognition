import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.io import read_video

class ActivityRecognitionModel:
    """
    A specialized model for activity recognition that:
      - Loads a model (already modified for multi-label classification)
      - Provides training and testing routines for short video clips.
      - Uses multi-label BCEWithLogitsLoss to allow for independent event predictions.
    """
    def __init__(self, model_architecture, model_path: str, num_labels: int, device=None):
        """
        Initialize the activity recognition model.
        
        Args:
            model_architecture: A callable that returns the model architecture.
            model_path (str): Path to the saved model state dictionary.
            num_labels (int): Number of event classes.
            device (optional): Device to run the model.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Load the model architecture and state dictionary.
        self.model = model_architecture().to(self.device)
        state_dict = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        
        # Modify the last layer for multi-label classification.
        if hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_labels).to(self.device)
        elif hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_labels).to(self.device)
        else:
            raise NotImplementedError("Model architecture does not support last layer modification.")
        
        self.model.to(self.device)
        # Define a loss function for independent multi-label classification.
        self.criterion = nn.BCEWithLogitsLoss()
        # Initialize an optimizer.
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

    def preprocess_input(self, video_path: str):
        """
        Preprocess the video clip for the activity recognition model.
        This is similar to the VideoLLM preprocessing.
        
        Args:
            video_path (str): Path to the video clip.
        
        Returns:
            Preprocessed video tensor.
        """
        # Read video frames.
        video_frames, _, _ = read_video(video_path, pts_unit='sec')
        video_frames = video_frames.permute(0, 3, 1, 2)  # (T, C, H, W)
        
        target_frames = 16
        T = video_frames.shape[0]
        if T > target_frames:
            indices = torch.linspace(0, T - 1, steps=target_frames).long()
            video_frames = video_frames[indices]
        elif T < target_frames:
            pad = video_frames[-1].unsqueeze(0).repeat(target_frames - T, 1, 1, 1)
            video_frames = torch.cat([video_frames, pad], dim=0)
        
        resize = transforms.Resize((224, 224))
        normalized_frames = []
        for frame in video_frames:
            frame = frame.float() / 255.0
            frame = resize(frame)
            frame = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])(frame)
            normalized_frames.append(frame)
        video_frames = torch.stack(normalized_frames)  # (target_frames, C, 224, 224)
        video_frames = video_frames.permute(1, 0, 2, 3).unsqueeze(0)  # (1, C, T, H, W)
        return video_frames.to(self.device)

    def forward(self, processed_input):
        """
        Forward pass through the model.
        """
        self.model.eval()
        with torch.no_grad():
            output = self.model(processed_input)
        return output

    def test(self, video_path: str):
        """
        Test the model on a given video clip.
        
        Args:
            video_path (str): Path to the video clip.
        
        Returns:
            Predicted probabilities for each event class.
        """
        self.model.eval()
        video_tensor = self.preprocess_input(video_path)
        with torch.no_grad():
            outputs = self.model(video_tensor)
        # For multi-label classification, apply sigmoid activation.
        probabilities = torch.sigmoid(outputs)
        return probabilities.cpu().numpy()

    def train_step(self, video_path: str, targets: torch.Tensor):
        """
        Perform one training step on a video clip:
          - Forward pass
          - Loss computation (using BCEWithLogitsLoss)
          - Backpropagation and optimizer step
        
        Args:
            video_path (str): Path to the video clip.
            targets (torch.Tensor): Ground truth labels of shape [batch, num_labels].
        
        Returns:
            The loss value for this training step.
        """
        self.model.train()
        video_tensor = self.preprocess_input(video_path)
        outputs = self.model(video_tensor)
        loss = self.criterion(outputs, targets.to(self.device))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def train_cycle(self, train_data, num_epochs: int = 10):
        """
        Run a full training cycle over the provided training data.
        
        Args:
            train_data: Iterable of tuples (video_path, targets)
            num_epochs (int): Number of training epochs.
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for video_path, targets in train_data:
                loss = self.train_step(video_path, targets)
                epoch_loss += loss
            avg_loss = epoch_loss / len(train_data)
            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def save_model(self, save_path: str):
        """
        Save the current state of the activity recognition model.
        
        Args:
            save_path (str): File path to save the model.
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Activity recognition model saved to {save_path}")