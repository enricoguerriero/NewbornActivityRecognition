def select_model(model_name, logger=None):
    if model_name == "timesformer":
        from models.baseline.timesformer import TimesformerModel
        model = TimesformerModel()
        logger.info("Selected model: Timesformer")
        return model
    elif model_name == "smolvlm":
        from models.LLMs.smolvlm import SmolVLMModel
        model = SmolVLMModel()
        logger.info("Selected model: SmolVLM")
        return model
    else:
        logger.error(f"Model {model_name} not available")
        
        
