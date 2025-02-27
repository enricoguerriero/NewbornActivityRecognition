import wandb
import torch

def select_model(model_name, logger=None):
    if model_name == "timesformer":
        from NewbornActivityRecognition.models.timesformer import TimesformerModel
        model = TimesformerModel()
        logger.info("Selected model: Timesformer")
        return model
    else:
        logger.error(f"Model {model_name} not available")
        
        
