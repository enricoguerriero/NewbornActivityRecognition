CONFIG = {
    # task
    "task": ["preprocessing", "test_untrained"],
    "model_name": "timesformer",
    
    # data creation
    "clip_length": 2,
    "frames_per_second": 8,
    "overlap": 0.5,
    "to_mp4": False,
    "transform": True,
    "target_size": (256, 256),
    "event_categories": ["Baby visible", "CPAP", "PPV", "Stimulation back/nates",
                                 "Stimulation extremities", "Stimulation trunk", "Suction"],
   
    # training parameters
    "batch_size": 16,
    "num_workers": 1,
    "epochs": 3,
    "learning_rate": 0.001,
    "num_classes": 7,
    "optimizer": "adam",
    "criterion": "bce",
    "threshold": 0.5,
    "device": "cuda",
    "momentum": 0.9,
    
    # wandb
    "wandb_project": "newborn-activity-recognition",
}
