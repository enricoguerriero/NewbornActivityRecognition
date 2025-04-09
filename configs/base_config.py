BASE_CONFIG = {
    
    # data creation
    "clip_length": 3,
    "frames_per_second": 8,
    "overlap": 2,
    "target_size": (256, 256),
    "event_categories": ["Baby visible", "Ventilation", "Stimulation", "Suction"],
   
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
    
    # wandb
    "wandb_project": "newborn-activity-recognition",
    
    # folder structure
    "video_folder": "data/videos",
    "annotation_folder": "data/annotations",
}

