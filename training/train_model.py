import torch
import torch.nn as nn
import torch.nn.functional as F
from synthai.src.tools import convert_readable

criterion = nn.CrossEntropyLoss()

def initialize_optimizer(model, learning_rate):
    return torch.optim.Adam(model.parameters(), lr=learning_rate)

def move_to_device(inputs, targets, device):
    inputs = inputs.to(device)
    targets = targets.to(device)
    return inputs, targets

def compute_loss(outputs, targets, criterion):
    outputs = outputs.view(-1, outputs.shape[-1])
    loss = criterion(outputs, targets.view(-1))
    return loss

def do_backprop(loss, optimizer):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def train_step(model, train_loader, optimizer, criterion, device):
    model.train()
    inputs, targets = next(iter(train_loader))
    inputs, targets = move_to_device(inputs, targets, device)
    outputs = model(inputs)
    loss = compute_loss(outputs, targets, criterion)
    do_backprop(loss, optimizer)
    return loss.item()

def val_step(model, val_loader, criterion, device):
    model.eval()
    with torch.no_grad():
        inputs, targets = next(iter(val_loader))
        inputs, targets = move_to_device(inputs, targets, device)
        outputs = model(inputs)
        generations = convert_readable(model.generate(inputs, 10))
        loss = compute_loss(outputs, targets, criterion)
        return loss.item(), generations

def train_model(model, train_loader, val_loader, config_file, device):
    config = config_file
    
    learning_rate = config.learning_rate 
    max_steps = config.max_steps 
    val_interval = config.val_interval

    optimizer = initialize_optimizer(model, learning_rate)
    train_losses = []
    val_losses = []
    generations = []
    step = 0 
    while True:
        train_loss = train_step(model, train_loader, optimizer, criterion, device)
        train_losses.append(train_loss)
        print(f"Step {step} | Training loss: {train_loss}")
        
        if step % val_interval == 0:
            val_loss, gen = val_step(model, val_loader, criterion, device)
            val_losses.append(val_loss)
            generations.append(gen)
            print(f"Step {step} | Validation loss: {val_loss}")
        
        step += 1
        if step >= max_steps: break
    


    with open("synthai/outputs/generations.txt", "w") as f:
        f.writelines(gen + "\n" for generated in generations for gen in generated)
        f.write("-" * 50 + "\n")

        

    return train_losses, val_losses

