from models.ResNetFeature import *
from utils import *
import os

def create_model(use_modulatedatt=False, use_fc=False, dropout=None, stage1_weights=False, dataset=None, weights_path=None, caffe=False, test=False):
    
    print('Loading Scratch ResNet 152 Feature Model.')
    resnet152 = ResNet(Bottleneck, [3, 8, 36, 3], use_modulatedatt=use_modulatedatt, use_fc=use_fc, dropout=None)
    
    if not test:

        #assert(caffe != stage1_weights)

        if caffe:
            print('Loading Caffe Pretrained ResNet 152 Weights.')
            resnet152 = init_weights(model=resnet152,
                                     weights_path=os.path.join(weights_path, 'final_model_checkpoint.pth'),
                                     caffe=True)
        elif stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 ResNet 152 Weights.' % dataset)
            resnet152 = init_weights(model=resnet152,
                                     weights_path=os.path.join(weights_path, 'final_model_checkpoint.pth'))
        else:
            print('No Pretrained Weights For Feature Model.')

    return resnet152
