from typing import Any
import lightning as L
import torch
from ..models.gpt_torch import GPT_Config, GPT_torch
from ..lightning_data.benchmarks_dm import BenchmarksConfig, BenchmarksDataModule
from lightning.pytorch.loggers import WandbLogger

model_config = GPT_Config(
    vocab_size = 50257,
    d_model = 8,
    cntx_len= 32,
    n_head= 2,
    n_layers= 2
)

train_config = BenchmarksConfig(
    data_dir = '/mnt/d/benchmarks/tiny_stories/',
    batch_size = 32,
    cntx = model_config.cntx_len,
    num_workers = 8,
    pin_memory = True,
    lr = 1e-3
)

class LitGPT(L.LightningModule):
    def __init__(self, model, config):
        super(LitGPT, self).__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters()
    
    def training_step(self, batch):
        x,y = batch
        loss,_ = self.model(x,y)
        self.log('train_loss', loss.item())
        return loss
    
    def validation_step(self, batch, dataloader_idx=None):
        x,y = batch
        loss,_ = self.model(x,y)
        self.log('valid_loss', loss.item())
        return loss
    
    def configure_optimizers(self):
        return torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)


model = GPT_torch(model_config)
config = train_config 

dm = BenchmarksDataModule(config)
lit_model = LitGPT(model, config)
wandb_logger = WandbLogger(project='SynthAI')
trainer = L.Trainer(accelerator='auto',devices=1, max_epochs=1,fast_dev_run=False,logger=wandb_logger)
trainer.fit(lit_model, dm)