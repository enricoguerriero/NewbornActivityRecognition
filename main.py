from config import CONFIG
from utils import setup_all_loggers, select_model
import logging
from data.dataset import VideoDataset
import os

def main():
    
    model_name = CONFIG["model_name"]
    tasks = CONFIG["task"]
    
    setup_all_loggers(model_name, tasks)
    
    logger = logging.getLogger(f'{model_name}_main')
    logger.info("--------------------------------")
    logger.info("...Starting the main function...")
    logger.info("--------------------------------")
    
    model = select_model(model_name)
    
    if "preprocessing" in tasks:
        model.transform = model.define_transformation(CONFIG["target_size"]) 
        model.preprocess_videos(set_name = "train", clip_length = CONFIG["clip_length"], frames_per_second = CONFIG["frames_per_second"],
                                overlap = CONFIG["overlap"], event_categories = ["event_categories"]) if "train" in tasks else None
        model.preprocess_videos(set_name = "validation") if "train" in tasks else None
        model.preprocess_videos("test") if "test" or "untrained_test" in tasks else None
        
    train_data = VideoDataset(os.path.join(model.output_folder, "train")) if "train" in tasks else None
    validation_data = VideoDataset(os.path.join(model.output_folder, "validation")) if "train" in tasks else None
    test_data = VideoDataset(os.path.join(model.output_folder, "test")) if "test" or "untrained_test" in tasks else None
        
    if "export" in tasks:
        pass
    
    if "train" in tasks:
        pass
    
    if "test" in tasks:
        pass
    
    if "untrained_test" in tasks:
        pass
    
    
    logger.info("...Exiting the main function...")
    

if __name__ == "__main__":
    main()