from models.ResNetFeature import *
from utils import *
import os

        
def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, weights_path=None, test=False, *args):
    
    print('Loading Scratch ResNet 10 Feature Model.')
    resnet10 = ResNet(BasicBlock, [1, 1, 1, 1], use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 10 Weights.' % dataset)
            resnet10 = init_weights(model=resnet10,
                                    weights_path=os.path.join(weights_path, 'final_model_checkpoint.pth'))
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet10
