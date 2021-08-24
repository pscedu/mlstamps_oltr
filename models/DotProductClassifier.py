import torch.nn as nn
from utils import *
import os

class DotProduct_Classifier(nn.Module):
    
    def __init__(self, num_classes=201, feat_dim=2048, *args):
        super(DotProduct_Classifier, self).__init__()
        self.fc = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x, *args):
        x = self.fc(x)
        return x, None
    
def create_model(feat_dim, num_classes=201, stage1_weights=False, dataset=None, weights_path= None, test=False, *args):
    print('Loading Dot Product Classifier.')
    clf = DotProduct_Classifier(num_classes, feat_dim)

    if not test:
        if stage1_weights:
            assert(dataset)
            print('Loading %s Stage 1 Classifier Weights.' % dataset)
            clf.fc = init_weights(model=clf.fc,
                                  weights_path=os.path.join(weights_path, 'final_model_checkpoint.pth'),
                                  classifier=True)
        else:
            print('Random initialized classifier weights.')

    return clf
