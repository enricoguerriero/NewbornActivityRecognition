from utils import setup_all_loggers, select_model, wandb_session, load_config, collate_fn
import logging
from data.clip_dataset import VideoDataset
from data.vlm_dataset import ClipDataset
import os
from argparse import ArgumentParser

def main():
    
    parser = ArgumentParser(description="Main function for video activity recognition.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the model to use.")
    parser.add_argument("--tasks", type=str, nargs='+', required=True, help="List of tasks to perform.")
    parser.add_argument("--load_model", type=str, default=None, help="Path to the model to load.")
    args = parser.parse_args()
    
    model_name = args.model_name
    tasks = args.tasks
    model_to_load = args.load_model
        
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
    
    if model_to_load:
        logger.info(f"Loading model from: {model_to_load}")
        model.load_model(model_to_load)
        logger.info(f"Model loaded: {model_to_load}")
    
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
                              event_categories = CONFIG["event_categories"],
                              processor = model.image_processor,
                              model_name = model_name,
                              tensor_folder = CONFIG["tensor_folder"],
                              set_name = "train") if "train" or "finetune" in tasks else None
    validation_data = VideoDataset(video_folder = os.path.join(CONFIG["video_folder"], "validation"),
                                   annotation_folder = os.path.join(CONFIG["annotation_folder"], "validation"),
                                   clip_length = CONFIG["clip_length"],
                                   frames_per_second = CONFIG["frames_per_second"],
                                   overlapping = CONFIG["overlap"],
                                   size = CONFIG["target_size"],
                                   tensors = True,
                                   event_categories = CONFIG["event_categories"],
                                   processor = model.image_processor,
                                   model_name = model_name,
                                   tensor_folder = CONFIG["tensor_folder"],
                                   set_name = "validation") if "train" or "finetune" in tasks else None
    test_data = VideoDataset(video_folder = os.path.join(CONFIG["video_folder"], "test"),
                              annotation_folder = os.path.join(CONFIG["annotation_folder"], "test"),
                              clip_length = CONFIG["clip_length"],
                              frames_per_second = CONFIG["frames_per_second"],
                              overlapping = CONFIG["overlap"],
                              size = CONFIG["target_size"],
                              tensors = True,
                              event_categories = CONFIG["event_categories"],
                              processor = model.image_processor,
                              model_name = model_name,
                              tensor_folder = CONFIG["tensor_folder"],
                              set_name = "test") if "test" or "untrained_test" in tasks else None
    
    train_loader = train_data.get_data_loader(batch_size = CONFIG["batch_size"],
                                              shuffle = True,
                                              num_workers = CONFIG["num_workers"]) if "train" in tasks or "finetune" in tasks else None
    validation_loader = validation_data.get_data_loader(batch_size = CONFIG["batch_size"], 
                                                        shuffle = False,
                                                        num_workers = CONFIG["num_workers"]) if "train" in tasks or "finetune" in tasks else None
    test_loader = test_data.get_data_loader(batch_size = CONFIG["batch_size"],
                                            shuffle = True,
                                            num_workers = CONFIG["num_workers"]) if "test" in tasks or "untrained_test" in tasks else None
    
    logger.info("...Data loaded...")
    
    if "train" in tasks:
        if CONFIG["criterion"] == "wbce":
            logger.info("...Computing weights for the loss function before training...")
            pos_weights, neg_weights = train_data.weight_computation()
            logger.info(f"Positive weights: {pos_weights}")
            logger.info(f"Negative weights: {neg_weights}")
        else:
            pos_weights, neg_weights = None, None
        logger.info("...Training the model...")
        history = model.train_model(train_loader = train_loader, 
                    optimizer = model.define_optimizer(CONFIG["optimizer"],
                                                       CONFIG["learning_rate"], 
                                                       CONFIG["momentum"]),
                    criterion = model.define_criterion(CONFIG["criterion"],
                                                       pos_weight=pos_weights.to(model.device), 
                                                       neg_weight=neg_weights.to(model.device)),
                    num_epochs = CONFIG["epochs"],
                    val_loader = validation_loader,
                    wandb = wandb,
                    early_stopping_patience = CONFIG["early_stopping_patience"],
                    early_stopping_delta = CONFIG["early_stopping_delta"])
        logger.info(f"History: {history}")

    if "test" in tasks:
        logger.info("...Testing the model...")
        history = model.test(dataloader = test_loader,
                   criterion = model.define_criterion(CONFIG["criterion"]), 
                   wandb = wandb) 
        logger.info(f"History: {history}")
        
    if "untrained_test" in tasks:
        logger.info("...Testing the model without knowledge...")
        history = model.test_without_knowledge(dataloader = test_loader, 
                                               questions = CONFIG["questions"],
                                               system_message = CONFIG["system_message"], 
                                               wandb = wandb)
        logger.info(f"History: {history}")


    if "finetune" in tasks:
        logger.info("...Fine-tuning the model...")
        logger.info(f"Fine-tuning with the following configuration: {CONFIG}")
        logger.info("Creating the dataset for fine-tuning...")
        train_dataset = ClipDataset(video_dataset=train_data,
                                    prompt = model.prompt_definition(system_message = CONFIG["system_message"],
                                                                    question = CONFIG["question"]),
                                    processor=model.processor)
        val_dataset = ClipDataset(video_dataset=validation_data, 
                                  prompt = model.prompt_definition(system_message = CONFIG["system_message"],
                                                                  question = CONFIG["question"]),
                                  processor=model.processor)
        logger.info("Fine-tuning the model...")
        model.train_model(train_dataset = train_dataset,
                          eval_dataset = val_dataset,
                          data_collator = collate_fn,
                          output_dir = CONFIG["output_dir"],
                          per_device_train_batch_size = CONFIG["batch_size"],
                          per_device_eval_batch_size = CONFIG["batch_size"],
                          gradient_accumulation_steps = CONFIG["gradient_accumulation_steps"],
                          num_train_epochs = CONFIG["num_train_epochs"],
                          learning_rate = CONFIG["learning_rate"],
                          logging_dir = CONFIG["logging_dir"],
                          evaluation_strategy = CONFIG["evaluation_strategy"],
                          save_strategy = CONFIG["save_strategy"],
                          fp16 = CONFIG["fp16"],
                          report_to = "wandb")


    logger.info("...Exiting the main function...")


if __name__ == "__main__":
    main()