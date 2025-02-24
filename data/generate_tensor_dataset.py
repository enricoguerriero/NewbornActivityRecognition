from data.utils.config import CONFIG
from data.classes.preprocessor import ClipPreprocessor

def main(logger):
    
    logger.debug("Generating dataset...")
    
    logger.debug(f"Preprocessing videos from {CONFIG['video_folder']}.")
    logger.debug(f"Using annotations from {CONFIG['annotation_folder']}.")
    
    preprocessor = ClipPreprocessor(
        video_folder=CONFIG["video_folder"],
        annotation_folder=CONFIG["annotation_folder"],
        output_folder=CONFIG["output_folder"],
        clip_length=CONFIG["clip_length"],
        frames_per_second=CONFIG["frames_per_second"],
        overlap=CONFIG["overlap"],
        target_size=CONFIG["target_size"],
        transform=CONFIG["transform"]
    )
    preprocessor.preprocess_all(logger)
    
    logger.debug("Dataset generated")
    

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger(__name__)
    main(logger)