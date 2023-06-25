from src.setup import preprocess
from src.unit_tests import test_transformer_block
import torch
import torch.nn as nn
import torch.nn.functional as F


def main():
    test_transformer_block()
    

    

    trainloader,valloader,testloader= preprocess(context_length=8,batch_size=2,shuffle=True)




if __name__ == '__main__':
    main()









