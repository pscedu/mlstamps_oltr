import os
import sys
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import logging
import numpy as np
import sklearn.model_selection

from utils import source_import
from shuffler.lib.utils import testUtils
from shuffler.interface.pytorch import datasets

import torch
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

# logging.basicConfig(level=1)

sys.path.append('/ocean/projects/hum180001p/prabha/mlstamps_oltr')
db_file = '/ocean/projects/hum180001p/etoropov/campaign6/crops/campaign3to6-6Kx4K.v7-croppedStamps.db'
rootdir = '/ocean/projects/hum180001p/shared/data'

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='/ocean/projects/hum180001p/prabha/mlstamps_oltr/config/stamps/stage_1.py', type=str)
parser.add_argument('--test', default=False, action='store_true')
parser.add_argument('--test_open', default=False, action='store_true')
parser.add_argument('--output_logits', default=False)
args = parser.parse_args()

test_mode = args.test
test_open = args.test_open

if test_open:
    test_mode = True

output_logits = args.output_logits

config = source_import(args.config).config
training_opt = config['training_opt']
# change
relatin_opt = config['memory']
dataset = training_opt['dataset']

if not os.path.isdir(training_opt['log_dir']):
    os.makedirs(training_opt['log_dir'])

pprint.pprint(config)

if not test_mode:

    sampler_defs = training_opt['sampler']
    if sampler_defs:
        sampler_dic = {'sampler': source_import(sampler_defs['def_file']).get_sampler(), 
                       'num_samples_cls': sampler_defs['num_samples_cls']}
    else:
        sampler_dic = None


    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    

    data = datasets.ObjectDataset(db_file,
                            rootdir=rootdir,
                            where_object="name NOT LIKE '%page%'",
                            mode='r',
                            used_keys=['image', 'objectid', 'name', 'num_instances'],
                            transform_group={'image': transform})

    dataset_size = data.__len__()
    print("\nTotal number of samples", dataset_size)

    validation_split = .2
    random_seed = 1
    shuffle = True
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    # if shuffle :
    #     np.random.seed(random_seed)
    #     np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    print("Training samples ", len(train_sampler))
    print("Validation samples ", len(valid_sampler))
    #print("Testing samples ", len(test_y))

    train_dataloader = torch.utils.data.DataLoader(data,
                                              batch_size=training_opt['batch_size'],
                                              num_workers=training_opt['num_workers'], sampler=train_sampler)

    test_dataloader = torch.utils.data.DataLoader(data,
                                              batch_size=training_opt['batch_size'],
                                              num_workers=training_opt['num_workers'], sampler=valid_sampler)


    training_model = model(config, train_dataloader, test=False)

    training_model.train()
    dataset.close()


else:

    warnings.filterwarnings("ignore", "(Possibly )?corrupt EXIF data", UserWarning)

    print('Under testing phase, we load training data simply to calculate training data number for each class.')

    # data = {x: dataloader.load_data(data_root=data_root[dataset.rstrip('_LT')], dataset=dataset, phase=x,
    #                                 batch_size=training_opt['batch_size'],
    #                                 sampler_dic=None, 
    #                                 test_open=test_open,
    #                                 num_workers=training_opt['num_workers'],
    #                                 shuffle=False)
    #         for x in ['train', 'test']}

    
    # training_model = model(config, data, test=True)
    # training_model.load_model()
    # training_model.eval(phase='test', openset=test_open)
    # #print(training_model)
    
    # if output_logits:
    #     training_model.output_logits(openset=test_open)
        
print('ALL COMPLETED.')
