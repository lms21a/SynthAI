import torch
from tools import read_json_config 
from synthai.models.gpt_reg import GPT_reg
from synthai.training.train_model import train_model
from setup_data import preprocess
TRAIN_FIGS = 'synthai/configs/train_configs.json'
MODEL_FIGS = 'synthai/configs/model_configs.json'

def main():
    # Read in configs
    model_configs = read_json_config(MODEL_FIGS)
    train_configs = read_json_config(TRAIN_FIGS)

    # Set Up Data 
    train_loader, val_loader,_ = preprocess(batch_size=model_configs.batch_size,
                                            context_length=model_configs.max_len,
                                            shuffle=model_configs.shuffle)

    # TODO: Adjust Preprocess to take in Config file instead of loose variables 

    # Set Up Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    model = GPT_reg(model_configs) 
    model = model.to(device)
    
    # Train Model
    train_losses, val_losses = train_model(model=model,
                                           train_loader=train_loader,
                                             val_loader=val_loader,
                                             config_file=train_configs,
                                             device=device)
    
    print("Training Complete")

if __name__ == '__main__':
    main()








