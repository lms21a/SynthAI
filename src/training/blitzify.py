import lightning as L
import torch
from .schedulers import CosWUScheduler

class Blitzify(L.LightningModule):
    def __init__(self, model, config):
        super(Blitzify, self).__init__()
        self.model = model
        self.config = config
        self.save_hyperparameters(ignore=['model'])
    
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
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
            fused=True,
            betas=(self.config.beta1, self.config.beta2)
        )
        scheduler = CosWUScheduler(
            optimizer=optimizer,
            lr=self.config.lr,
            warmup_steps=self.config.warmup_steps,
            lr_decay_steps=self.config.total_steps
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }