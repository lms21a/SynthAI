from typing import Any
import lightning as L
import torch
from .models.LlamaKinda import LlamaKinda
from .models.components import LlamaKindaArgs
from .lightning_data.benchmarks_dm import BenchmarksConfig, BenchmarksDataModule
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from .training.blitzify import Blitzify

torch.set_float32_matmul_precision('high')
wandb_logger = WandbLogger(project='SynthAI')
checkpoint_callback = ModelCheckpoint(dirpath='checkpoints', save_last=True, every_n_train_steps=25000)

model_config = LlamaKindaArgs(
    dim = 512,
    n_layers = 8,
    n_qheads = 8,
    n_kvheads = 8,
    max_seq_len = 768,
    multiplier = 4
)
train_config = BenchmarksConfig(
    data_dir = '/mnt/d/benchmarks/openweb/',
    batch_size = 32,
    cntx = model_config.max_seq_len,
    num_workers = 8,
    pin_memory = True,
    lr = 8e-4,
    warmup_steps= 1500,
    total_steps= 3e5,
    weight_decay = 1e-5,
    compile = True,
    shuffle=True
)
trainer = L.Trainer(
    fast_dev_run=False, 
    accelerator='auto',
    devices=1,
    max_steps=train_config.total_steps,
    limit_val_batches=0.5,
    val_check_interval = 5000,
    precision='bf16-mixed', 
    logger=wandb_logger, 
    accumulate_grad_batches=8, 
    # gradient_clip_val=1.0, # Fused AdamW does it internally
    num_sanity_val_steps=10,
    callbacks=[checkpoint_callback]
)

model = torch.compile(LlamaKinda(model_config)) if train_config.compile else LlamaKinda(model_config)
config = train_config 

dm = BenchmarksDataModule(config)
lit_model = Blitzify(model, config)

trainer.fit(lit_model, dm)