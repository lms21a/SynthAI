import torch
from unit_tests import run_tests
import torch.nn as nn
import torch.nn.functional as F
from setup_data import preprocess
def main():
    train_loader, val_loader, test_loader = preprocess(context_length=8,batch_size=64,shuffle=True)
    run_tests()
    tx,ty = next(iter(train_loader))
    print(tx.shape)
    print(ty.shape)


if __name__ == '__main__':
    main()









