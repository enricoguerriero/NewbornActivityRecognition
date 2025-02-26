from utils.config import CONFIG
from utils.logger import create_logger
from utils.utils import select_model, wandb_session, define_optimizer, define_criterion

def main():
    
    task = CONFIG["task"] 
    gen_data = CONFIG["generate_data"]
    model_name = CONFIG["model_name"]
    logger = create_logger("main", f"logs/{model}_{'_'.join(str(t) for t in task)}.log")
    
    logger.info("-----------------------")
    logger.info(" ... Starting main ... ")
    logger.info("-----------------------")
    
    if gen_data:
        from data.generate_tensor_dataset import main as generate_data
        generate_data(logger)

    logger.info("Loading dataset...")
    from data.load_dataset import main as load_dataset
    train_data_loader, val_data_loader, test_data_loader = load_dataset(logger)
    logger.info("Dataset loaded")
    
    model = select_model(model_name, logger)
    
    if "train" in task:
        wandb = wandb_session(CONFIG["wandb_project"] + "_train", CONFIG)
        logger.info("Training model...")
        model.train_model(train_loader = train_data_loader,
                          optimizer = define_optimizer(CONFIG["optimizer"], model, CONFIG["learning_rate"], CONFIG["momentum"]),
                          criterion = define_criterion(CONFIG["criterion"]),
                          num_epochs = CONFIG["epochs"],
                          val_loader = val_data_loader,
                          logger = logger,
                          wandb = wandb)
        logger.info("Model trained")
    
    if "test" in task:
        wandb = wandb_session(CONFIG["wandb_project"] + "_test", CONFIG)
        logger.info("Testing model...")
        model.test(test_data_loader,
                   criterion = define_criterion(CONFIG["criterion"]),
                   logger = logger,
                   wandb = wandb)
        logger.info("Model tested")    
    


if __name__ == "__main__":
    main()