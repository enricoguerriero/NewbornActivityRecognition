from config import CONFIG
from utils import setup_all_loggers, select_model, wandb_session
import logging
from data.dataset import VideoDataset
import os
import json
from data.preprocess_data import preprocess_videos

def main():
    
    model_name = CONFIG["model_name"]
    tasks = CONFIG["tasks"]
    
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
        logger.info("...Preprocessing the data...")
        for set in ["test"]: # ["train", "validation", "test"]
            preprocess_videos(model_name, CONFIG["video_folder"], CONFIG["annotation_folder"], CONFIG["output_folder"],
                              CONFIG["target_size"], model.image_processor, set, CONFIG["clip_length"],
                              CONFIG["frames_per_second"], CONFIG["overlap"], CONFIG["event_categories"])
        logger.info("...Data preprocessed...")

    logger.info("...Loading data...")
        
    # train_data = VideoDataset(os.path.join(model.output_folder, "train")) if "train" in tasks else None
    # validation_data = VideoDataset(os.path.join(model.output_folder, "validation")) if "train" in tasks else None
    test_data = VideoDataset(os.path.join(model.output_folder, "test")) if "test" or "untrained_test" in tasks else None
    
    # train_loader = train_data.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"]) if "train" in tasks else None
    # validation_loader = validation_data.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"]) if "train" in tasks else None
    test_loader = test_data.get_data_loader(CONFIG["batch_size"], CONFIG["num_workers"]) if "test" or "untrained_test" in tasks else None
    
    logger.info("...Data loaded...")
    
    # create the labels.json file
    if "labeling" in tasks:
        logger.info("...Creating the labels file...")
        # structure of the json:
        # {
        #     "video_name": {
        #         "clip_number": {
        #             "ground_truth": "label",
        #             "model_name": "predicted_label",
        #             ...},
        #         ...},
        #     ...}
        with open('labels.json', 'w') as f:
            json.dump({}, f)
        # each frame is saved like this
        # clip_data = {
        #         'frames': frames_tensor,
        #         'labels': clip_labels,
        #         'video_path': video_path,
        #         'clip_index': clip_index,
        #         'clip_start_time': clip_start_time,
        #         'clip_length': clip_length,
        #         'sampling_rate': frames_per_second
        #     }
        # the labels are saved in clip_labels
        # fill the file with the ground truth labels
        for clip in os.listdir(os.path.join(model.output_folder, "test")):
            clip_path = os.path.join(model.output_folder, "test", clip)
            with open(os.path.join(clip_path), 'r') as f:
                clip_data = json.load(f)
                clip_labels = clip_data['labels']
                clip_name = clip.split('.')[0]
                with open('labels.json', 'r') as f:
                    labels_data = json.load(f)
                labels_data[clip_name] = {}
                for i in range(len(clip_labels)):
                    labels_data[clip_name][i] = {"ground_truth": clip_labels[i]}
                with open('labels.json', 'w') as f:
                    json.dump(labels_data, f)
                
        logger.info("...Labels file created...")
                
    logger.info("...Testing the model without knowledge...")
    history, new_labels = model.test_without_knowledge(test_loader, questions = None, wandb = wandb)
    logger.info(f"History: {history}")
    
    if "labeling" in tasks:
        logger.info("...Labeling the data...")
        # open the file json in read mode
        with open("labels.json", 'r') as f:
            labels_data = json.load(f)
        logger.info(f"Labels loaded: {labels_data}")
        # add the new labels to the file json
        for clip_name in new_labels.keys():
            labels_data[clip_name][model_name] = new_labels[clip_name]
        # save the file json
        with open("labels.json", 'w') as f:
            json.dump(labels_data, f)
        logger.info("...Data labeled...")


    logger.info("...Exiting the main function...")


if __name__ == "__main__":
    main()