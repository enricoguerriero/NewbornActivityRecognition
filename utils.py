import logging
import wandb
import importlib
import torch

def load_config(model_name):
    try:
        config_module = importlib.import_module(f"configs.{model_name}")
        return config_module.CONFIG
    except ModuleNotFoundError:
        raise ValueError(f"No config found for model '{model_name}'")


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
    elif model_name == "llavavideo_classifier":
        from models.llavavideo_classifier import VideoLlavaClassifier
        return VideoLlavaClassifier()
    else:
        from models.promptLLM import PromptLLMModel
        engine = select_engine(model_name)
        return PromptLLMModel(engine)
    
def select_engine(engine_name):
    """Function to select an engine."""
    if engine_name == "smolvlm":
        from engines.smolvlm import SmolVLMEngine
        return SmolVLMEngine()
    elif engine_name == "smolvlm256":
        from engines.smolvlm256 import SmolVLM256Engine
        return SmolVLM256Engine()
    # elif engine_name == "smolvlm500":
    #     from engines.smolvlm500 import SmolVLM500Engine
    #     return SmolVLM500Engine()
    elif engine_name == "llava_video":
        from engines.llavavideo import VideoLLavaEngine
        return VideoLLavaEngine()
    elif engine_name == "janus":
        from engines.janus7B import JanusProEngine
        return JanusProEngine()
    # elif engine_name == "valley":
    #     from engines.valley import ValleyEngine
    #     return ValleyEngine()
    else:
        raise ValueError(f"Engine {engine_name} not available")
    
def wandb_session(project_name, model_name, config):
    wandb.init(project=project_name + " - " + model_name, config=config)
    return wandb

def collate_fn(batch):
    for item in batch:
        print(f'item keys: {item.keys()}', flush=True)
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "pixel_values": torch.stack([item["pixel_values"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }