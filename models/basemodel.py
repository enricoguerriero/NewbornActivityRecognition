import torch.nn as nn
import logging
import os
from tqdm import tqdm
from torchvision.transforms import transforms
import cv2
import numpy as np
import torch
# from .wBCE import WeightedBCELoss

class BaseVideoModel(nn.Module):
    """
    An abstract base class for video models.
    Provides a common interface for training, inference, and last-layer modifications.
    """
    def __init__(self, device = "cuda", model_name: str = "baseModel"):
        super(BaseVideoModel, self).__init__()
        self.model_name = model_name
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
        
    def define_optimizer(self, optimizer_name, learning_rate, momentum = None, weight_decay = None):
        """
        Defines the optimizer for the model.
        By now you can choose between Adam and SGD.
        """
        if optimizer_name == "adam":
            optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.parameters()),
                                         lr=learning_rate)
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.parameters()),
                                        lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        elif optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()),
                                          lr=learning_rate, weight_decay=weight_decay)
        else:
            raise ValueError(f"Optimizer {optimizer_name} not available")
        return optimizer
    
    def define_criterion(self, criterion_name, pos_weight=None, neg_weight=None):
        """
        Defines the criterion for the model.
        By now you can choose between BCE and CrossEntropy.
        """
        if criterion_name == "bce":
            criterion = torch.nn.BCEWithLogitsLoss()
        elif criterion_name == "crossentropy":
            criterion = torch.nn.CrossEntropyLoss()
        elif criterion_name == "wbce":
            # criterion = WeightedBCELoss(pos_weight=pos_weight, neg_weight=neg_weight)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        else:
            raise ValueError(f"Criterion {criterion_name} not available")
        return criterion
