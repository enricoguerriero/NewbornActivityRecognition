from data.classes.dataset import PreprocessedClipDataset
from utils.config import CONFIG

def main(logger):

    logger.debug("Preparing train dataset...")
    train_dataset = PreprocessedClipDataset(preprocessed_folder=CONFIG["train_output_folder"])
    logger.debug("Train dataset instanciated")
    logger.debug("Preparing validation dataset...")
    val_dataset = PreprocessedClipDataset(preprocessed_folder=CONFIG["validation_output_folder"])
    logger.debug("Validation dataset instanciated")
    logger.debug("Preparing test dataset...")
    test_dataset = PreprocessedClipDataset(preprocessed_folder=CONFIG["test_output_folder"])
    logger.debug("Test dataset instanciated")
    
    if CONFIG["to_mp4"]:
        logger.debug("Exporting train clips to MP4...")
        train_dataset.export_all_clips_to_mp4(CONFIG["train_export_folder"], export_fps=CONFIG["export_fps"], logger = logger)
        logger.debug("Exporting validation clips to MP4...")
        val_dataset.export_all_clips_to_mp4(CONFIG["validation_export_folder"], export_fps=CONFIG["export_fps"], logger = logger)
        logger.debug("Exporting test clips to MP4...")
        test_dataset.export_all_clips_to_mp4(CONFIG["test_export_folder"], export_fps=CONFIG["export_fps"], logger = logger)
        logger.debug("Clips exported")
        
    logger.debug("Loading data loaders...")
    train_data_loader = train_dataset.get_data_loader(batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"])
    logger.debug("Dataset loaded")
    for batch in train_data_loader:
        input, target = batch
        logger.debug(f"Input shape: {input.shape}")
        logger.debug(f"Target shape: {target.shape}")
        break
    val_data_loader = val_dataset.get_data_loader(batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"])
    logger.debug("Dataset loaded")
    for batch in val_data_loader:
        input, target = batch
        logger.debug(f"Input shape: {input.shape}")
        logger.debug(f"Target shape: {target.shape}")
        break
    test_data_loader = test_dataset.get_data_loader(batch_size=CONFIG["batch_size"], num_workers=CONFIG["num_workers"])
    logger.debug("Dataset loaded")
    for batch in test_data_loader:
        input, target = batch
        logger.debug(f"Input shape: {input.shape}")
        logger.debug(f"Target shape: {target.shape}")
        break
    
    return train_data_loader, val_data_loader, test_data_loader
    

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    main(logger)