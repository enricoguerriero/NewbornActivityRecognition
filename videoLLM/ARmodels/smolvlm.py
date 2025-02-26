from transformers import AutoModel
import torch
import torch.nn as nn
import torch.optim as optim
from videoLLM.ARmodels.activity_recognition_model import ActivityRecognitionModel

class SmolVLMActivityRecognitionModel(ActivityRecognitionModel):
    def __init__(self, num_labels: int, model_name: str = "smolvlm", device=None):
        """
        Initialize the smolvlm model from Hugging Face for activity recognition.

        Args:
            num_labels (int): Number of event classes.
            model_name (str): Hugging Face model identifier for smolvlm.
            device (optional): Device to run the model on.
        """
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load the pretrained smolvlm model from Hugging Face.
        model = AutoModel.from_pretrained(model_name)
        
        # Modify the last layer to support multi-label classification.
        if hasattr(model, 'classifier'):
            in_features = model.classifier.in_features
            model.classifier = nn.Linear(in_features, num_labels).to(self.device)
        elif hasattr(model, 'fc'):
            in_features = model.fc.in_features
            model.fc = nn.Linear(in_features, num_labels).to(self.device)
        else:
            raise NotImplementedError("SmolVLM model does not support last layer modification.")
        
        self.model = model.to(self.device)
        self.criterion = nn.BCEWithLogitsLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)
