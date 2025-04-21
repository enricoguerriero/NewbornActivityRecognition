from .base_config import BASE_CONFIG

CONFIG = BASE_CONFIG.copy()

CONFIG.update({
    "clip_length": 2,
    "frames_per_second": 6,
    "overlap": 1,
    "batch_size": 1,
    
    "output_dir": "models/saved/llavavideo_classifier",
})