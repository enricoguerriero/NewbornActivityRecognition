import torch.nn as nn
import torch
from abc import ABC
import wandb
from torchvision import transforms

class GenericModel(nn.Module, ABC):
    """
    Generic base class for video activity recognition models.
    """

    def __init__(self, device=None):
        """
        Initializes the base model.
        
        Args:
            device (str): Device to run on (e.g., 'cuda' or 'cpu').
        """
        super(GenericModel, self).__init__()
        # Set device (fallback to CPU if CUDA is not available)
        self.device = torch.device(device) if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_name = None


    def wandb_session(self, project_name, config):
        wandb.init(project=project_name + " - " + self.model_name, config=config)
        return wandb
    
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
    
    def define_transformation(self, target_size):
        """
        Defines the transformation for the model.
        """
        transform = transforms.Compose([
            transforms.ToPILImage(),             # Convert numpy array to PIL Image.
            transforms.Resize(target_size),         # Resize the image.
        ])
        return transform