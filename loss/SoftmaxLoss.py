import torch.nn as nn

def create_loss ():
    print('Loading Softmax Loss.')
    #return nn.NLLLoss()
    return nn.CrossEntropyLoss()

