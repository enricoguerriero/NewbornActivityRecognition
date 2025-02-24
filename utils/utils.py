import wandb

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