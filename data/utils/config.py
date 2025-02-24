CONFIG = {
    "train_video_folder": "data/videos/train",
    "train_annotation_folder": "data/annotations/train",
    "train_output_folder": "data/preprocessed/train",
    "train_export_folder": "data/exported/train",
    
    "val_video_folder": "data/videos/val",
    "val_annotation_folder": "data/annotations/val",
    "val_output_folder": "data/preprocessed/val",
    "val_export_folder": "data/exported/val",
    
    "test_video_folder": "data/videos/test",
    "test_annotation_folder": "data/annotations/test",
    "test_output_folder": "data/preprocessed/test",
    "test_export_folder": "data/exported/test",
    
    "to_mp4": True,
    "export_fps": None,
    "batch_size": 64,
    "num_workers": 4,
    "clip_length": 2,
    "frames_per_second": 10,
    "overlap": 0.5,
    "target_size": (224, 224),
    "transform": None,
}