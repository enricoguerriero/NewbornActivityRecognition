from utils import setup_all_loggers, select_model, wandb_session, load_config
import logging
from data.clip_dataset import VideoDataset
import os
from argparse import ArgumentParser

def main():
    
    parser = ArgumentParser(description="Main function for video activity recognition.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--tasks", type=str, nargs='+', required=True, help="List of tasks to perform.")
    args = parser.parse_args()
    
    model_name = args.model_name
    tasks = args.tasks
        
    setup_all_loggers(model_name, tasks)
    
    CONFIG = load_config(model_name)
    
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
    
    logger.info("...Loading data...")
        
    train_data = VideoDataset(video_folder = os.path.join(CONFIG["video_folder"], "train"),
                              annotation_folder = os.path.join(CONFIG["annotation_folder"], "train"),
                              clip_length = CONFIG["clip_length"],
                              frames_per_second = CONFIG["frames_per_second"],
                              overlapping = CONFIG["overlap"],
                              size = CONFIG["target_size"],
                              tensors = True,
                              event_categories = CONFIG["event_categories"]) if "train" in tasks else None
    validation_data = VideoDataset(video_folder = os.path.join(CONFIG["video_folder"], "validation"),
                                   annotation_folder = os.path.join(CONFIG["annotation_folder"], "validation"),
                                   clip_length = CONFIG["clip_length"],
                                   frames_per_second = CONFIG["frames_per_second"],
                                   overlapping = CONFIG["overlap"],
                                   size = CONFIG["target_size"],
                                   tensors = True,
                                   event_categories = CONFIG["event_categories"]) if "train" in tasks else None
    test_data = VideoDataset(video_folder = os.path.join(CONFIG["video_folder"], "test"),
                              annotation_folder = os.path.join(CONFIG["annotation_folder"], "test"),
                              clip_length = CONFIG["clip_length"],
                              frames_per_second = CONFIG["frames_per_second"],
                              overlapping = CONFIG["overlap"],
                              size = CONFIG["target_size"],
                              tensors = True,
                              event_categories = CONFIG["event_categories"]) if "test" or "untrained_test" in tasks else None
    
    train_loader = train_data.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"]) if "train" in tasks else None
    validation_loader = validation_data.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"]) if "train" in tasks else None
    test_loader = test_data.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"]) if "test" or "untrained_test" in tasks else None
    
    logger.info("...Data loaded...")
        
    if "export" in tasks:
        pass
    
    if "train" in tasks:
        logger.info("...Training the model...")
        history = model.train_model(train_loader = train_loader, 
                    optimizer = model.define_optimizer(CONFIG["optimizer"], CONFIG["learning_rate"], CONFIG["momentum"]),
                    criterion = model.define_criterion(CONFIG["criterion"]),
                    num_epochs = CONFIG["epochs"],
                    val_loader = validation_loader,
                    wandb = wandb)
        logger.info(f"History: {history}")

    if "test" in tasks:
        logger.info("...Testing the model...")
        history = model.test(dataloader = test_loader,
                   criterion = model.define_criterion(CONFIG["criterion"]), 
                   wandb = wandb) 
        logger.info(f"History: {history}")
        
    if "test_untrained" in tasks:
        logger.info("...Testing the model without knowledge...")
        history = model.test_without_knowledge(data_loader = test_loader, 
                                               questions = CONFIG["question"],
                                               system_message = CONFIG["system_message"], 
                                               wandb = wandb)
        logger.info(f"History: {history}")


    logger.info("...Exiting the main function...")


if __name__ == "__main__":
    main()