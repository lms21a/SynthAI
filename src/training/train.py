from typing import Any
import lightning as L
import torch
from ..models.gpt_torch import GPT_Config, GPT_torch
from ..lightning_data.benchmarks_dm import BenchmarksConfig, BenchmarksDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint

torch.set_float32_matmul_precision('high')

model_config = GPT_Config(
    vocab_size = 50257,
    d_model = 512,
    cntx_len= 1024,
    n_head= 8,
    n_layers= 8,
    dropout_p= 0.1
)

train_config = BenchmarksConfig(
    data_dir = '/mnt/d/benchmarks/tiny_stories/',
    batch_size = 24,
    cntx = model_config.cntx_len,
    num_workers = 8,
    pin_memory = True,
    lr = 8e-4
)

class Blitzify(L.LightningModule):
    def __init__(self, model, config):
        super(Blitzify, self).__init__()
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
lit_model = Blitzify(model, config)
lit_model.load_from_checkpoint('checkpoints/last-v1.ckpt')
wandb_logger = WandbLogger(project='SynthAI')
checkpoint_callback = ModelCheckpoint(dirpath='checkpoints/', save_on_train_epoch_end=True, filename = 'Tiny-Stories-{epoch}', save_last=True)
trainer = L.Trainer(
    accelerator='auto',
    devices=1,
    max_epochs=10,
    # max_steps=20, 
    val_check_interval = 5000,
    limit_val_batches=.1,
    precision='bf16-mixed', 
    fast_dev_run=False, 
    logger=wandb_logger, 
    accumulate_grad_batches=6, 
    callbacks=[checkpoint_callback])
trainer.fit(lit_model, dm, ckpt_path='checkpoints/last-v1.ckpt')