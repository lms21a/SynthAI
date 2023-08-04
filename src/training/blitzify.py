import lightning as L
from .schedulers import CustomCosineAnnealingWarmupScheduler
import torch

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
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.lr)
        scheduler = CustomCosineAnnealingWarmupScheduler(
            optimizer,
            warmup_steps=self.config.warmup_steps,
            total_steps=self.config.total_steps,
            cycles=self.config.cycles
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
