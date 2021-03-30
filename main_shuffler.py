import os
import sys
import argparse
import pprint
from data import dataloader
from run_networks import model
import warnings
import logging
import sklearn.model_selection

from utils import source_import
from shuffler.lib.utils import testUtils
from shuffler.interface.pytorch import datasets

import torch
from torchvision import transforms
# logging.basicConfig(level=1)

sys.path.append('/ocean/projects/pscstaff/rajanie/MLStamps/long-tail/mlstamps-oltr')
db_file = '/ocean/projects/pscstaff/rajanie/MLStamps/long-tail/OpenLongTailRecognition-OLTR/campaign6-6Kx4K.v6.db'
rootdir = '/ocean/projects/pscstaff/rajanie/MLStamps/long-tail/OpenLongTailRecognition-OLTR/data'

parser = argparse.ArgumentParser()
parser.add_argument('--config', default='/ocean/projects/pscstaff/rajanie/MLStamps/long-tail/mlstamps-oltr/config/stamps/stage_1.py', type=str)
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

    num_samples = data.__len__()
    print("\nTotal number of samples", num_samples)
    #print([data[x]['name'] for x in range(0,data.__len__()-1)])

    x = [data[x]['image'] for x in range(0,data.__len__()-1)]
    print(x)
    y = [data[x]['name'] for x in range(0,data.__len__()-1)]
    

    train_x, test_val_x, train_y, test_val_y = sklearn.model_selection.train_test_split([data[x]['image'] for x in range(0,data.__len__()-1)], [data[x]['name'] for x in range(0,data.__len__()-1)], test_size=0.3, random_state=1, stratify=[data[x]['name'] for x in range(0,data.__len__()-1)])
    val_x, test_x, val_y, test_y = sklearn.model_selection.train_test_split(test_val_x, test_val_y, test_size=0.5, random_state=1, stratify=test_val_y)

    print("Training samples ", len(train_y))
    print("Validation samples ", len(val_y))
    print("Testing samples ", len(test_y))

    
    

    final_train_data = torch.utils.data.DataLoader(data,
                                              batch_size=training_opt['batch_size'],
                                              shuffle=True,
                                              num_workers=training_opt['num_workers'])


    training_model = model(config, final_train_data, test=False)

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
