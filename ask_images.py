import logging
from utils import select_model
import os
from PIL import Image

def main():
    
    MODEL_NAME = "janus7b"
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("demo")
    logger.info("--------------------------------")
    logger.info("...Starting the demo function...")
    logger.info("--------------------------------")
    
    model = select_model(MODEL_NAME)
    logger.info(f"Model initialized: {MODEL_NAME}")
    
    IMAGES_FOLDER = "imgs"
    
    QUESTION = "this is a medical emergency call and this is an image of the scene/patient. Can you please describe what you see and what you think have happened?"
    
    for image_file in os.listdir(IMAGES_FOLDER):
        image_path = os.path.join(IMAGES_FOLDER, image_file)
        if os.path.isfile(image_path):
            logger.info(f"Processing image: {image_path}")
            image = Image.open(image_path).convert("RGB")
            predictions = model.prompt_engine.answer_questions([image], [QUESTION], seed=42, temperature=0.2)
            logger.info(f"Predictions for {image_file}: {predictions}")
            # write predictions to a file
            with open(f"{image_file}_predictions.txt", "w") as f:
                f.write(str(predictions))
            logger.info(f"Predictions saved to {image_file}_predictions.txt")
        else:
            logger.warning(f"Skipping non-file: {image_path}")
    
    
    
    
if __name__ == "__main__":
    main()