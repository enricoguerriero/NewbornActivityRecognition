import logging

def setup_logger(name, filename):
    """Function to setup a logger."""
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    file_handler = logging.FileHandler("logs/" + filename + ".log")        
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger

def setup_all_loggers(model_name: str, tasks: list):
    """Function to setup all loggers."""
    setup_logger(f'{model_name}_main', model_name)
    for task in tasks:
        setup_logger(f'{model_name}_{task}', model_name)
        
        
def select_model(model_name):
    """Function to select a model."""
    if model_name == "timesformer":
        from models.timesformer import TimesformerModel
        return TimesformerModel()
    else:
        from models.promptLLM import PromptLLMModel
        engine = select_engine(model_name)
        return PromptLLMModel(engine)
    
def select_engine(engine_name):
    """Function to select an engine."""
    if engine_name == "smolvlm":
        from engines.smolvlm import SmolVLMEngine
        return SmolVLMEngine()
    else:
        raise ValueError(f"Engine {engine_name} not available")