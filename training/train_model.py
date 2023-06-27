import torch
import torch.nn as nn
import torch.nn.functional as F
from setup_data import preprocess
# Training Parameters
MAX_STEPS = 10
LR = 0.001
OPTIMIZER = torch.optim.Adam
LOSS = nn.CrossEntropyLoss()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
while True:
    inputs,targets = next(iter(train_loader))
    inputs.to(device)
    targets.to(device)
    outputs = model(inputs)

