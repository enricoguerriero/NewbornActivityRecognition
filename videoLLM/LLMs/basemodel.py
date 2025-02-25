import abc
import torch
import torch.nn as nn

class BaseLLMModel(abc.ABC):
    """
    A generic LLM wrapper for video clip analysis that provides common functionality:
      - Preprocess video input
      - Forward pass (with optional prompt)
      - Testing the model
      - Modifying the last layer for multi-label classification
      - Saving the model locally
      
    Subclasses should implement the video preprocessing and forward pass.
    """
    def __init__(self, model, device=None):
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

    @abc.abstractmethod
    def preprocess_input(self, video_path: str):
        """
        Process a raw video clip into a tensor.
        
        Args:
            video_path (str): Path to the video clip.
        
        Returns:
            A tensor representing the video clip.
        """
        pass

    @abc.abstractmethod
    def forward(self, processed_input, prompt=None):
        """
        Perform a forward pass through the model.
        
        Args:
            processed_input: Preprocessed video tensor.
            prompt (optional): Text prompt for additional context.
        
        Returns:
            Raw model output.
        """
        pass

    def test_model(self, video_path: str, prompt=None):
        """
        Evaluate the model on the provided video clip (and optional prompt).
        
        Args:
            video_path (str): Path to the video clip.
            prompt (optional): Text prompt.
        
        Returns:
            The raw model output.
        """
        self.model.eval()
        video_tensor = self.preprocess_input(video_path)
        with torch.no_grad():
            output = self.forward(video_tensor, prompt)
        return output

    def modify_last_layer(self, num_labels: int):
        """
        Modify the model's last layer to be a multi-label classification head.
        This implementation looks for either a `classifier` or an `fc` attribute.
        
        Args:
            num_labels (int): Number of independent event classes.
        
        Returns:
            The modified model.
        """
        if hasattr(self.model, 'classifier'):
            in_features = self.model.classifier.in_features
            self.model.classifier = nn.Linear(in_features, num_labels).to(self.device)
        elif hasattr(self.model, 'fc'):
            in_features = self.model.fc.in_features
            self.model.fc = nn.Linear(in_features, num_labels).to(self.device)
        else:
            raise NotImplementedError("Last layer modification not implemented for this model architecture.")
        return self.model

    def save_model(self, save_path: str):
        """
        Save the current model's state dictionary locally.
        
        Args:
            save_path (str): File path for saving the model.
        """
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

