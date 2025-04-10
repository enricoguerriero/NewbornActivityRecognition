from .base_config import BASE_CONFIG

CONFIG = BASE_CONFIG.copy()
CONFIG.update({    
    # specific configurations for TimesFormer
    "target_size": (224, 224),
    
    # training parameters
    "batch_size": 16,
    "num_workers": 1,
    "epochs": 10,
    "learning_rate": 0.001,
    "optimizer": "adam",
    "criterion": "bce",
    "threshold": 0.5,
    "device": "cuda",
    "momentum": 0.9,
})