from models.basemodel import BaseVideoModel
import torch.nn as nn

class TimesformerModel(BaseVideoModel):
    def __init__(self, backbone, num_classes):
        """
        :param backbone: A pre-built Timesformer backbone that processes video tensors.
        :param num_classes: Number of activity classes.
        """
        super(TimesformerModel, self).__init__()
        self.backbone = backbone  # Assume backbone outputs feature vectors.
        # Simple classifier head: you can replace it with a multi-layer head if needed.
        self.classifier = nn.Linear(backbone.out_features, num_classes)
    
    def forward(self, x):
        # x is expected to be a tensor of shape (num_frames, channels, height, width) or batched accordingly.
        features = self.backbone(x)
        return self.classifier(features)
    
    def modify_last_layer(self, new_layer_config):
        self.classifier = new_layer_config
        return self
