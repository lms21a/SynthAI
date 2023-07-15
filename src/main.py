import torch
from datasets import load_dataset
from .tools import read_json_config,visualize 
from .models.gpt_torch import GPT_Config, GPT_torch 
from .training.train_model import Trainer 
from .data_pipeline import Preprocessor
TRAIN_FIGS = 'configs/train_configs.json'

def main():
    # Read in configs
    model_configs = GPT_Config() 
    train_configs = read_json_config(TRAIN_FIGS)

    # Set Up Data 
    
    ds = load_dataset(
        path='bigcode/the-stack-dedup',
        data_dir='data/python',
        save_infos=True,
        split='train',
        num_proc=8
    )
    preprocessor = Preprocessor(
        ds,
        batch_size=train_configs.batch_size,
        cntx_len=model_configs.cntx_len,
        shuffle=train_configs.shuffle,
        test_size=train_configs.test_size)

    train_loader, val_loader = preprocessor.preprocess()

   
    
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







