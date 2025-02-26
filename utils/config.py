CONFIG = {
    # folders
    "train_video_folder": "data/videos/train",
    "train_annotation_folder": "data/annotations/train",
    "train_output_folder": "data/preprocessed/train",
    "train_export_folder": "data/exported/train",
    
    "validation_video_folder": "data/videos/validation",
    "validation_annotation_folder": "data/annotations/validation",
    "validation_output_folder": "data/preprocessed/validation",
    "validation_export_folder": "data/exported/validation",
    
    "test_video_folder": "data/videos/test",
    "test_annotation_folder": "data/annotations/test",
    "test_output_folder": "data/preprocessed/test",
    "test_export_folder": "data/exported/test",
    
    # data generation
    "T": 0.2,
    "W": 0,

    # data loading
    "to_mp4": True,
    "export_fps": None,
    "batch_size": 16,
    "num_workers": 1,
    "clip_length": 2,
    "frames_per_second": 5,
    "overlap": 0.5,
    "target_size": (224, 224),
    "transform": None,
    
    # training parameters
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