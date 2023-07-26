import torch
from datasets import load_dataset
from .models.gpt_torch import GPT_Config, GPT_torch 
from .training.train import Trainer
from .training.train_configs import Dev_Train
from .data_pipeline import Preprocessor

def main():
    # Read in configs
    model_configs = GPT_Config()
    train_configs = Dev_Train() 



    # Set Up Data 
    # Add Cache dir
    ds = load_dataset(
        path='bigcode/the-stack-dedup',
        data_dir='data/python',
        save_infos=True,
        split='train',
        num_proc=8
    )
    # # Preprocess Data
    # preprocessor = Preprocessor(
    #     dataset=ds,
    #     batch_size=train_configs.batch_size,
    #     cntx_len=model_configs.cntx_len,
    #     shuffle=train_configs.shuffle,
    #     test_size=train_configs.test_size)

    # train_loader, val_loader = preprocessor.preprocess()

   # TODO: Set Up text datasets for processing; pass in string, use locator to get outputs 
    # Set Up Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device} for training")
    model = GPT_torch(model_configs)
    # Model Statistics and Device Agnostics
    model.print_model_size()
    model = model.to(device)
    model.count_model_memory()

    # Train Model
    trainer = Trainer(model, train_configs, device, train_loader, val_loader)
    trainer.train()
    print("Training Complete")

if __name__ == '__main__':
    main()







