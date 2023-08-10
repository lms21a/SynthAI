import tiktoken
import torch
import torch.nn.functional as F 
from .models.gpt_torch import GPT_Config, GPT_torch
import lightning as L
from .lightning_data.benchmarks_dm import BenchmarksConfig, BenchmarksDataModule
from .tools import convert_readable

torch.set_float32_matmul_precision('high')

model_config = GPT_Config(
    vocab_size = 50257,
    d_model = 512,
    cntx_len= 1024,
    n_head= 8,
    n_layers= 8,
    dropout_p= 0.1
)

model = GPT_torch(model_config)

model.eval()
enc = tiktoken.get_encoding('gpt2')
checkpoint = torch.load('checkpoints/last-v1.ckpt')

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
config = None
model = Blitzify(model, config).load_from_checkpoint('checkpoints/last-v2.ckpt')
model.eval()
model.cuda()
enc = tiktoken.get_encoding('gpt2')

with torch.inference_mode():
    while True:
        start = input('Enter a prompt: ')
        if start == 'exit':
            break
        start_ids = enc.encode(start)

        x = (torch.tensor(start_ids, dtype = torch.long, device = 'cuda')[None,...])
        y = model.model.generate(x, 500, 1)
        print(convert_readable(y)[0])
        print('-'*10)