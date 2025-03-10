from transformers import AutoImageProcessor, TimesformerForVideoClassification
from torch import nn
from models.basemodel import BaseVideoModel    

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
        
        self.input_type = "tensors"

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