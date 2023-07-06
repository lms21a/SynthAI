import torch
import torch.nn as nn
from ..tools import convert_readable
import os
from tqdm.auto import tqdm
from ..training.schedulers import CustomCosineAnnealingWarmupScheduler
class Trainer:
    def __init__(self, model, config_file, device, train_loader, val_loader):
        self.model = model
        self.config = config_file
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = self.initialize_optimizer()
        self.scheduler = self.initialize_scheduler() if self.config.scheduler else None
        self.train_losses = []
        self.val_losses = []
        self.generations = []

    def initialize_optimizer(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

    def initialize_scheduler(self):
            if self.config.scheduler == 'cosine':
                return CustomCosineAnnealingWarmupScheduler(
                    self.optimizer,
                    warmup_steps=self.config.warmup_steps,
                    total_steps=self.config.max_steps,
                    cycles=self.config.cycles if hasattr(self.config, 'cycles') else 0.5
                )
            else:
                pass

    def move_to_device(self, inputs, targets):
        inputs = inputs.to(self.device)
        targets = targets.to(self.device)
        return inputs, targets

    def compute_loss(self, outputs, targets):
        outputs = outputs.view(-1, outputs.shape[-1])
        loss = self.criterion(outputs, targets.view(-1))
        return loss

    def do_backprop(self, loss, step):
        loss = loss / self.config.grad_accum_steps  # Normalize our loss (if averaged)
        loss.backward()
        
        if (step+1) % self.config.grad_accum_steps == 0:
            if self.config.clip_value is not None:
                # Clip gradients to prevent them from exploding
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.clip_value)
            
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad()
    
    def save_checkpoint(self, step):
            """Save a training checkpoint."""

            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'step': step,
                'train_losses': self.train_losses,
                'val_losses': self.val_losses
                # Add any other state information you want to preserve
            }
            
            # You can organize your checkpoints in subdirectories if you want
            checkpoint_dir = os.path.join("checkpoints", f"checkpoint_{step}.pt")
            torch.save(checkpoint, checkpoint_dir)
            print(f"Saved checkpoint at step {step}.")

    def forward_pass(self,inputs,targets):
        with torch.autocast(device_type='cuda', enabled=self.config.mixed_precision):
            outputs = self.model(inputs)
            loss = self.compute_loss(outputs, targets)
        return outputs, loss

    def train_step(self, step):
        self.model.train()
        inputs, targets = next(iter(self.train_loader))
        inputs, targets = self.move_to_device(inputs, targets)

        # Call the forward_pass function to get outputs and loss
        outputs, loss = self.forward_pass(inputs, targets)

        self.do_backprop(loss, step)
        return loss.item()

    def val_step(self):
        self.model.eval()
        with torch.no_grad():
            inputs, targets = next(iter(self.val_loader))
            inputs, targets = self.move_to_device(inputs, targets)
            outputs, loss = self.forward_pass(inputs, targets)
            generations = convert_readable(self.model.generate(inputs, 100))
            return loss.item(), generations

    def train(self):
        step = 0 

        # Initialize a tqdm progress bar
        pbar = tqdm(total=self.config.max_steps, desc="Training Progress")

        while True:
            train_loss = self.train_step(step)
            self.train_losses.append(train_loss)

            if step % self.config.val_interval == 0:
                val_loss, gen = self.val_step()
                self.val_losses.append(val_loss)
                self.generations.append(gen)
                pbar.set_postfix({"Training Loss": train_loss, "Validation Loss": val_loss})

            # If checkpoint_interval is not 0, save a checkpoint every checkpoint_interval steps
            if self.config.checkpoint_interval and step % self.config.checkpoint_interval == 0:
                self.save_checkpoint(step)
                
            step += 1
            pbar.update(1)  # update the progress bar

            if step >= self.config.max_steps: 
                pbar.close()
                break

        with open("outputs/generations.txt", "w") as f:
            f.writelines(gen + "\n" for generated in self.generations for gen in generated)

        return self.train_losses, self.val_losses