import torch
from .components import LearnFormerBlock, LearnFormerArgs

# Adapted from Easy attention: A simple self-attention mechanism for Transformers
# Essentially, we use the attention score as a learnable parameter
# https://arxiv.org/pdf/2308.12874.pdf 
class LearnFormer(torch.nn.Module):
    def __init__(self, args: LearnFormerArgs):
        super().__init__()
        self.embed = torch.nn.Embedding(args.vocab_size, args.dim)
        self.learn_pos = torch.nn.Linear(args.cntx, args.dim)
        self.lm_head = torch.nn.Linear(args.dim, args.vocab_size)

        # tie weights
        self.embed.weight = self.lm_head.weight
        
        self.blocks = torch.nn.ModuleDict({f'block_{i}': LearnFormerBlock(args) for i in range(args.nlayers)})
        
    def forward(self, x, y = None):
        l = x.size(1)
        pos = torch.arange(0, l, dtype=torch.float32, device=x.device)
        x = self.embed(x) + self.learn_pos(pos)
        for block in self.blocks.values():
            x = block(x)
        
        logits = self.lm_head(x)
        if y is not None:
            loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            return loss
        else:
            return logits

# Test 
args = LearnFormerArgs()
dummy = torch.randint(0, args.vocab_size, (args.bsz, args.cntx))
target = torch.randint(0, args.vocab_size, (args.bsz, args.cntx))
print(LearnFormer(args)(dummy, target))