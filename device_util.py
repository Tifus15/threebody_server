import torch 
import os

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")
DEVICE = torch.device("cpu")
ROOT_PATH = os.path.dirname(os.path.abspath(__file__))

def init():
    global ROOT_PATH
    global DEVICE

def DEVICETensor(*args):
    return torch.FloatTensor(*args).to(DEVICE)

def CPUTensor(*args):
    return torch.FloatTensor(*args)