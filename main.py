from config import CONFIG
from utils import setup_all_loggers, select_model, wandb_session
import logging
from data.dataset import VideoDataset
import os
import wandb

def main():
    
    model_name = CONFIG["model_name"]
    tasks = CONFIG["task"]
    
    setup_all_loggers(model_name, tasks)
    
    logger = logging.getLogger(f'{model_name}_main')
    logger.info("--------------------------------")
    logger.info("...Starting the main function...")
    logger.info("--------------------------------")
    
    model = select_model(model_name)
    
    wandb = wandb_session(CONFIG["wandb_project"], model_name, CONFIG)
    
    if "preprocessing" in tasks:
        logger.info("...Preprocessing videos...")
        model.transform = model.define_transformation(CONFIG["target_size"]) 
        model.preprocess_videos(set_name = "train", clip_length = CONFIG["clip_length"], frames_per_second = CONFIG["frames_per_second"],
                                overlap = CONFIG["overlap"], event_categories = ["event_categories"]) if "train" in tasks else None
        model.preprocess_videos(set_name = "validation") if "train" in tasks else None
        model.preprocess_videos("test") if "test" or "untrained_test" in tasks else None
        
    logger.info("...Loading data...")
        
    train_data = VideoDataset(os.path.join(model.output_folder, "train")) if "train" in tasks else None
    validation_data = VideoDataset(os.path.join(model.output_folder, "validation")) if "train" in tasks else None
    test_data = VideoDataset(os.path.join(model.output_folder, "test")) if "test" or "untrained_test" in tasks else None
    
    train_loader = train_data.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"]) if "train" in tasks else None
    validation_loader = validation_data.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"]) if "train" in tasks else None
    test_loader = test_data.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"]) if "test" or "untrained_test" in tasks else None
    
    logger.info("...Data loaded...")
        
    if "export" in tasks:
        pass
    
    if "train" in tasks:
        pass
    
    if "test" in tasks:
        pass
    
    if "untrained_test" in tasks:
        logger.info("...Testing the model without knowledge...")
        model.test_without_knowledge(test_loader, questions = None, wandb = wandb)
    
    
    logger.info("...Exiting the main function...")
    

if __name__ == "__main__":
    main()