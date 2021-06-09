#dataset = MNIST(...) assert len(dataset) == 60000 
#for training and 10000 for test assert that each 
# datapoint has shape [1,28,28] or [728] depending 
# on how you choose to format assert that all labels 
# are represented
import sys, os
import argparse
import torch
from torch import nn, optim
import matplotlib.pyplot as plt
print("torch version")
print(torch.__version__)


def datalen():
    print(torch.__version__)
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    train_set = torch.load("../data/processed/train.pt")
    test_set = torch.load("../data/processed/test.pt")


    return len(train_set.dataset),len(test_set.dataset)

def datashape():
    # Change directory
    dir_path = os.path.dirname(os.path.realpath(__file__))
    os.chdir(dir_path)
    
    train_set = torch.load("../data/processed/train.pt")
    print("HEY")
    print(train_set.dataset)
    print(torch.__version__)
    images, lables = next(iter(train_set))
    print(images.size())


    

    return images.size()


def test_datalen():
    len_train, len_test = datalen()
    assert len_train == 60000 
    assert len_test == 10000




def test_datashape():
    x = datashape()
    assert list(x) == [64,1,28,28]