import wandb
import torch

def select_model(model_name, logger=None):
    if model_name == "timesformer":
        from baseline.models.timesformer import TimesformerModel
        model = TimesformerModel()
        logger.info("Selected model: Timesformer")
        return model
    else:
        logger.error(f"Model {model_name} not available")
        
        
def wandb_session(project_name, config):
    wandb.init(project=project_name, config=config)
    return wandb

def define_optimizer(optimizer_name, model, learning_rate, momentum):
    """
    Defines the optimizer for the model.
    By now you can choose between Adam and SGD.
    """
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "sgd":
        optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    else:
        raise ValueError(f"Optimizer {optimizer_name} not available")
    return optimizer

def define_criterion(criterion_name):
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