from config import CONFIG
from utils import setup_all_loggers, select_model, wandb_session
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
    
    logger.info(f"Tasks: {tasks}")
    logger.info(f"Model: {model_name}")
    model = select_model(model_name)
    logger.info(f"Model initialized: {model_name}")
    
    wandb = wandb_session(CONFIG["wandb_project"], model_name, CONFIG)
    logger.info(f"Wandb session started: {CONFIG['wandb_project']} - {model_name}")
    
    logger.debug(f'Configuration: {CONFIG}')
    
    if "preprocessing" in tasks:
        logger.info("...Preprocessing videos...")
        model.transform = model.define_transformation(CONFIG["target_size"]) 
        logger.debug(f"Transform: {model.transform}")
        model.preprocess_videos(set_name = "train", clip_length = CONFIG["clip_length"], frames_per_second = CONFIG["frames_per_second"],
                                overlap = CONFIG["overlap"], event_categories = CONFIG["event_categories"]) if "train" in tasks else None
        model.preprocess_videos(set_name = "validation", clip_length = CONFIG["clip_length"], frames_per_second = CONFIG["frames_per_second"],
                                overlap = CONFIG["overlap"], event_categories = CONFIG["event_categories"]) if "train" in tasks else None
        model.preprocess_videos(set_name = "test", clip_length = CONFIG["clip_length"], frames_per_second = CONFIG["frames_per_second"],
                                overlap = CONFIG["overlap"], event_categories = CONFIG["event_categories"]) if "test" or "untrained_test" in tasks else None
        
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
        logger.info("...Training the model...")
        model.train_model(train_loader = train_loader, 
                    optimizer = model.define_optimizer(CONFIG["optimizer"], CONFIG["learning_rate"], CONFIG["momentum"]),
                    criterion = model.define_criterion(CONFIG["criterion"]),
                    num_epochs = CONFIG["epochs"],
                    val_loader = validation_loader,
                    wandb = wandb)
    
    if "test" in tasks:
        pass
    
    if "test_untrained" in tasks:
        logger.info("...Testing the model without knowledge...")
        model.test_without_knowledge(test_loader, questions = None, wandb = wandb)
    
    
    logger.info("...Exiting the main function...")
    

if __name__ == "__main__":
    main()