# The purpose of this script is to try models on a small subset of the test set with 0 shot
from utils import select_model
import logging
from data.clip_dataset import VideoDataset
import os
from argparse import ArgumentParser
from data.clip_subset import VideoSubsetDataset
from torch.utils.data import DataLoader
import json

def main():
    
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("demo")
    logger.info("--------------------------------")
    logger.info("...Starting the demo function...")
    logger.info("--------------------------------")
    
    # clip indexes for demo
    CLIP_INDEXES = [80, 142, 220, 324, 412, 644, 691, 841, 1048, 1090, 1395, 1426, 1493, 1532, 1643, 1813, 1887, 1921, 2040, 2197, 2379, 2446]    
    
    # WORKING JUST WITH 1 SEC CLIP LENGTH - OVERLAP
    
    parser = ArgumentParser()
    parser.add_argument("model_name", type=str, help="Model name")
    
    args = parser.parse_args()
    model_name = args.model_name
    
    model = select_model(model_name)
    
    logger.info(f"Model initialized: {model_name}")
    
    VIDEO_FOLDER = "data/videos/demo"
    ANNOTATION_FOLDER = "data/annotations/demo"
    OUTPUT_FOLDER = "data/outputs/demo"
    
    CLIP_LENGTH = 3 
    OVERLAP = 2
    SIZE = 256
    FPS = 5
    EVENT_CATEGORIES = ["Baby visible", "Ventilation", "Stimulation", "Suction"]
    
    QUESTIONS = ["There is supposed to be a baby / doll on the table in front of the camera. Is there a baby / doll on the table?",
                 "If there is a baby / doll on the table, is it receiving ventilation? It is supposed to wear a mask covering the nose and the mouth, is the baby / the doll wearing it?",
                 "If there is a baby / doll on the table, is it receiving stimulation? It is supposed to be stimulated by rubbing with a movement that goes up and down either the back or the trunk of the baby / doll, is it being stimulated?",
                 "If there is a baby / doll on the table, is it receiving suction? There should be a thin tube that goes into the mouth of the baby / doll, is it there? Is it being used?"]
    logger.info("Creating the dataset...")
    
    dataset = VideoDataset(
        video_folder=VIDEO_FOLDER,
        annotation_folder=ANNOTATION_FOLDER,
        clip_length=CLIP_LENGTH,
        overlapping=OVERLAP,
        size=SIZE,
        frames_per_second=FPS,
        tensors=False,
        event_categories=EVENT_CATEGORIES
    )
    
    video_idx = 0
    clip_subset = VideoSubsetDataset(
        full_dataset=dataset,
        video_idx=video_idx,
        clip_indexes=CLIP_INDEXES
    )
    logger.info(f"Subset dataset created with {len(clip_subset)} clips.")
    
    subset_loader = DataLoader(
        clip_subset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    logger.info(f"Subset DataLoader created with {len(subset_loader)} batches.")
    
    # structure of the json:
    # [
    #     clip_number: {
    #         "ground_truth": "label",
    #         "model_name": "predicted_label",
    #     ...},
    # ...]
    if not os.path.exists("demo_labels.json"):
        with open("demo_labels.json", "w") as f:
            json.dump({}, f)
        logger.info("Labels file created.")
        labels = {}
        for i, clip in enumerate(clip_subset):
            label = clip['labels']
            clip_number = i   
            labels[clip_number] = {}
            labels[clip_number]["ground_truth"] = label.tolist()
        with open("demo_labels.json", "w") as f:
            json.dump(labels, f)
        logger.info("Labels file filled with ground truth labels.")
    else:
        with open("demo_labels.json", "r") as f:
            labels = json.load(f)
        logger.info("Labels loaded.")
        
    logger.info("Starting the demo...")
    history, new_labels = model.test_without_knowledge(subset_loader, questions=QUESTIONS, wandb=None)
    logger.info(f"History: {history}")
    logger.info(f"New labels: {new_labels}")
    logger.info(f"Labels: {labels}")
    
    # update labels with new labels
    for i, clip in enumerate(clip_subset):
        clip_number = i
        labels[clip_number]["model_name"] = new_labels[clip["clip_name"]]
    with open("demo_labels.json", "w") as f:
        json.dump(labels, f)
    logger.info("Labels file updated with model predictions.")
    logger.info("Demo finished.")
    logger.info("--------------------------------")
    
if __name__ == "__main__":
    main()