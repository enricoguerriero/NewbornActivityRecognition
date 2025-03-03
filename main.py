from utils.config import CONFIG
from utils.logger import create_logger
from utils.utils import select_model
from data.dataset import PreprocessedClipDataset

def main():
    
    task = CONFIG["task"] 
    model_name = CONFIG["model_name"]
    logger = create_logger("main", f"logs/{model_name}_{'_'.join(str(t) for t in task)}.log")
    
    logger.info("-----------------------")
    logger.info(" ... Starting main ... ")
    logger.info("-----------------------")
    
    model = select_model(model_name, logger)
    
    if CONFIG["generate_data"]:
        logger.info("Preprocessing clips...")
        transform = model.define_transformation(CONFIG["target_size"]) if CONFIG["transform"] else None
        from data.preprocessor import ClipPreprocessor
        preprocessor = ClipPreprocessor(clip_length = CONFIG["clip_length"],
                                        frames_per_second = CONFIG["frames_per_second"],
                                        overlap = CONFIG["overlap"],
                                        event_categories = CONFIG["event_categories"],
                                        transform = transform,
                                        processor = model.processor)
        for set_name in CONFIG["set_to_generate"]:
            preprocessor.preprocess_all(video_folder = CONFIG[f"{set_name}_video_folder"], 
                                        annotation_folder = CONFIG[f"{set_name}_annotation_folder"],
                                        output_folder = CONFIG[f"{set_name}_output_folder"], 
                                        logger = logger)
        logger.info("Clips preprocessed")

    train_dataset = PreprocessedClipDataset(CONFIG["train_output_folder"]) if "train" in task else None
    val_dataset = PreprocessedClipDataset(CONFIG["validation_output_folder"]) if "train" in task else None
    test_dataset = PreprocessedClipDataset(CONFIG["test_output_folder"]) if "test" in task else None
    
    if CONFIG["to_mp4"]:
        logger.info("Exporting clips to MP4...")
        train_dataset.export_all_clips_to_mp4(CONFIG["train_export_folder"],
                                              label_list = CONFIG["event_categories"],
                                              logger = logger) if "train" in task else None
        val_dataset.export_all_clips_to_mp4(CONFIG["validation_export_folder"], 
                                            label_list = CONFIG["event_categories"], 
                                            logger = logger) if "train" in task else None
        test_dataset.export_all_clips_to_mp4(CONFIG["test_export_folder"], 
                                             label_list = CONFIG["event_categories"], 
                                             logger = logger) if "test" in task else None
        logger.info("Clips exported to MP4")
    
    if "train" in task:
        logger.info("Loading data...")
        train_data_loader = train_dataset.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"])
        val_data_loader = val_dataset.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"])
        wandb = model.wandb_session(CONFIG["wandb_project"] + "_train", CONFIG)
        logger.info("Training model...")
        model.train_model(train_loader = train_data_loader,
                          optimizer = model.define_optimizer(CONFIG["optimizer"], CONFIG["learning_rate"], CONFIG["momentum"]),
                          criterion = model.define_criterion(CONFIG["criterion"]),
                          num_epochs = CONFIG["epochs"],
                          val_loader = val_data_loader,
                          logger = logger,
                          wandb = wandb)
        logger.info("Model trained")
    
    if "test" in task:
        wandb = model.wandb_session(CONFIG["wandb_project"] + "_test", CONFIG)
        logger.info("Testing model...")
        model.test(test_dataset,
                   criterion = model.define_criterion(CONFIG["criterion"]),
                   logger = logger,
                   wandb = wandb)
        logger.info("Model tested")  
        
    if "test untrained" in task:
        wandb = model.wandb_session(CONFIG["wandb_project"] + "_test_untrained", CONFIG)
        logger.info("Testing untrained model...")
        model.test_without_knowledge(test_dataset,
                   logger = logger,
                   wandb = wandb)
        logger.info("Model tested")  
    


if __name__ == "__main__":
    main()