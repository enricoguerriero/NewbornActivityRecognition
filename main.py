import argparse
from utils.config import CONFIG
from utils.logger import create_logger
from utils.utils import select_model, wandb_session

def main():
    
    parser = argparse.ArgumentParser(description='Task to run')
    parser.add_argument('task', type=str, default='', help='Task to run')
    parser.add_argument('--model', type=str, default='timesformer', help='Model to use')
    parser.add_argument('--generate_data', type=str, default=None, help='Generate new data') 
    
    args = parser.parse_args()
    
    task = args.task
    gen_data = args.generate_data
    model_name = args.model
    CONFIG["model_name"] = model_name
    logger = create_logger("main", f"logs/{task}.log")
    
    logger.info("---------------------")
    logger.info("... Starting main ...")
    logger.info("---------------------")
    
    if gen_data:
        from data.generate_tensor_dataset import main as generate_data
        generate_data(logger)

    logger.info("Loading dataset...")
    from data.load_dataset import main as load_dataset
    train_data_loader, val_data_loader, test_data_loader = load_dataset(logger)
    logger.info("Dataset loaded")
    
    model = select_model(model_name, logger)
    
    if task == "train":
        wandb = wandb_session(CONFIG["wandb_project"] + "_train", CONFIG)
        logger.info("Training model...")
        model.train_model(train_loader = train_data_loader,
                          optimizer = model.optimizer,
                          criterion = model.criterion,
                          num_pochs = CONFIG["num_epochs"],
                          learning_rate = CONFIG["learning_rate"],
                          momentum = CONFIG["momentum"],
                          val_loader = val_data_loader,
                          logger = logger,
                          wandb = wandb)
        logger.info("Model trained")
    
    if task == "test":
        wandb = wandb_session(CONFIG["wandb_project"] + "_test", CONFIG)
        logger.info("Testing model...")
        model.test(test_data_loader,
                   criterion = model.criterion,
                   logger = logger,
                   wandb = wandb)
        logger.info("Model tested")    
    


if __name__ == "__main__":
    main()