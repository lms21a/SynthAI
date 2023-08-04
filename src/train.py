from typing import Any
import lightning as L
import torch
from .models.gpt_torch import GPT_Config, GPT_torch
from .lightning_data.benchmarks_dm import BenchmarksConfig, BenchmarksDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from .training.blitzify import Blitzify

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

model = GPT_torch(model_config)
config = train_config 

dm = BenchmarksDataModule(config)
lit_model = Blitzify(model, config)

wandb_logger = WandbLogger(project='SynthAI')

checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', save_last=True)
trainer = L.Trainer(
    accelerator='auto',
    devices=1,
    max_steps=20,
    val_check_interval = 5000,
    precision='bf16-mixed', 
    fast_dev_run=False, 
    logger=wandb_logger, 
    accumulate_grad_batches=6, 
    callbacks=[checkpoint_callback])

trainer.fit(lit_model, dm)