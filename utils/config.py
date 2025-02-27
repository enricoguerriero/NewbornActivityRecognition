CONFIG = {
    # task
    "task": ["train", "test"],
    "generate_data": True,
    "model_name": "timesformer",
    
    # folders
    "train_video_folder": "data/videos/train/",
    "train_annotation_folder": "data/annotations/train/",
    "train_output_folder": "data/preprocessed/train/",
    "train_export_folder": "data/exported/train/",
    
    "validation_video_folder": "data/videos/validation/",
    "validation_annotation_folder": "data/annotations/validation/",
    "validation_output_folder": "data/preprocessed/validation/",
    "validation_export_folder": "data/exported/validation/",
    
    "test_video_folder": "data/videos/test/",
    "test_annotation_folder": "data/annotations/test/",
    "test_output_folder": "data/preprocessed/test/",
    "test_export_folder": "data/exported/test/",
    
    "specific_folder" : "timesformer",
    
    # data loading
    "data_loading": True,
    "to_mp4": False,
    "export_fps": None,
    "batch_size": 16,
    "num_workers": 1,
    "clip_length": 2,
    "frames_per_second": 8,
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