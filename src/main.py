import torch
from unit_tests import run_tests
import torch.nn as nn
import torch.nn.functional as F
from setup_data import preprocess, convert_readable
from synthai.models.gpt_reg import GPT_reg
# Hyperparameters
MAX_LEN = 8
BATCH_SIZE = 64
SHUFFLE = True
VOCAB_SIZE = 50257
D_MODEL = 64
NUM_HEADS = 4
NUM_LAYERS = 2

def main():
    train_loader, val_loader,_= preprocess(context_length=MAX_LEN, batch_size=BATCH_SIZE, shuffle=SHUFFLE)
    #run_tests()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT_reg(
        vocab_size=VOCAB_SIZE,
        d_model=D_MODEL,
        num_heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        max_len=MAX_LEN
    )

    MAX_STEPS = 10
    LR = 0.001
    OPTIMIZER = torch.optim.Adam(model.parameters(), lr=LR)
     
    train_losses = []
    val_losses = []
    generations = []
    criterion = nn.CrossEntropyLoss()
    val_interval = 5 
    steps = 0
    while True:

        model.train()
        inputs,targets = next(iter(train_loader))
        inputs.to(device)
        targets.to(device)
        outputs = model(inputs)
        # reshape outputs to match targets
        outputs = outputs.view(-1, outputs.shape[-1])
        loss = criterion(outputs,targets.view(-1)) 
        OPTIMIZER.zero_grad()
        loss.backward()
        OPTIMIZER.step()
        train_losses.append(loss.item())
        print(f"Step {steps} | Training loss: {loss.item()}")

        if steps % val_interval == 0 and steps != 0:
            model.eval()
            with torch.no_grad():
                inputs,targets = next(iter(val_loader))
                inputs.to(device)
                targets.to(device)
                outputs = model(inputs)
                generations.append(convert_readable(model.generate(inputs,10)))
                outputs = outputs.view(-1, outputs.shape[-1])
                loss = criterion(outputs,targets.view(-1)) 
                val_losses.append(loss.item())
                # Generate from the model
                print(f"Step {steps} | Validation loss: {loss.item()}")
        
        steps += 1
        if steps == MAX_STEPS: break


    with open("synthai/outputs/generations.txt", "w") as f:
        f.writelines(gen + "\n" for generated in generations for gen in generated)
        f.write("-" * 50 + "\n")

if __name__ == '__main__':
    main()








