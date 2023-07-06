import torch
from .tools import read_json_config,visualize 
from .models.gpt_torch import GPT_Config, GPT_torch 
from .training.train_model import Trainer 
from .setup_data import preprocess
TRAIN_FIGS = 'configs/train_configs.json'

def main():
    # Read in configs
    model_configs = GPT_Config() 
    train_configs = read_json_config(TRAIN_FIGS)

    # Set Up Data 
    train_loader, val_loader,_ = preprocess(batch_size=train_configs.batch_size,
                                            context_length=model_configs.cntx_len,
                                            shuffle=train_configs.shuffle_loaders)

    # TODO: WANDB Logging Needs better integration
    
    # Set Up Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    model = GPT_torch(model_configs)
    # Model Statistics and Device Agnostics
    model.print_model_size()
    model = model.to(device)
    model.count_model_memory()

    # Train Model
    trainer = Trainer(
        model = model,
        config_file = train_configs,
        device = device,
        train_loader = train_loader,
        val_loader = val_loader
    )
    train_losses, val_losses = trainer.train()
    
    visualize(train_losses,val_losses) 
    print("Training Complete")

if __name__ == '__main__':
    main()







