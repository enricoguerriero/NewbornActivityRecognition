def select_model(model_name, logger=None):
    if model_name == "timesformer":
        from models.baseline.timesformer import TimesformerModel
        model = TimesformerModel()
        logger.info("Selected model: Timesformer")
        return model
    else:
        logger.error(f"Model {model_name} not available")
        
        
