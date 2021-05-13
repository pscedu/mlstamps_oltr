import os
import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm
from utils import *
import time
import numpy as np
import warnings
import pdb
np.set_printoptions(threshold=np.inf)
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics as skmetrics
from sklearn.metrics import classification_report
from pytorch_lightning import metrics

class model ():
    
    def __init__(self, config, full_data, data, val_data, test=False):

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.config = config
        self.training_opt = self.config['training_opt']
        self.memory = self.config['memory']
        self.full_data = full_data
        self.data = data
        self.val_data = val_data
        self.test_mode = test
        
        # Initialize model
        self.init_models()

        # Under training mode, initialize training steps, optimizers, schedulers, criterions, and centroids
        if not self.test_mode:

            # If using steps for training, we need to calculate training steps 
            # for each epoch based on actual number of training data instead of 
            # oversampled data number 
            print('Using steps for training.')
            #self.training_data_num = len(self.data['train'].dataset)
            self.training_data_num = len(self.data)
            self.epoch_steps = int(self.training_data_num  \
                                   / self.training_opt['batch_size'])

            # Initialize model optimizer and scheduler
            print('Initializing model optimizer.')
            self.scheduler_params = self.training_opt['scheduler_params']
            self.model_optimizer, self.model_optimizer_scheduler = self.init_optimizers(self.model_optim_params_list)
            self.init_criterions()
            if self.memory['init_centroids']:
                self.criterions['FeatureLoss'].centroids.data = \
                    self.centroids_cal(self.data)
            
        # Set up log file
        self.log_file = os.path.join(self.training_opt['log_dir'], 'log.txt')
        print("Logging into ", self.log_file)
        # if os.path.isfile(self.log_file):
        #     os.remove(self.log_file)
        
    def init_models(self, optimizer=True):

        networks_defs = self.config['networks']
        self.networks = {}
        self.model_optim_params_list = []
        
        for key, val in networks_defs.items():

            # Networks
            def_file = val['def_file']
            model_args = list(val['params'].values())
            model_args.append(self.test_mode)

            self.networks[key] = source_import(def_file).create_model(*model_args)
            self.networks[key] = nn.DataParallel(self.networks[key]).to(self.device)
            
            if 'fix' in val and val['fix']:
                print('Freezing feature weights except for modulated attention weights (if exist).')
                for param_name, param in self.networks[key].named_parameters():
                    # Freeze all parameters except self attention parameters
                    if 'modulatedatt' not in param_name and 'fc' not in param_name:
                        param.requires_grad = False

            # Optimizer list
            optim_params = val['optim_params']
            self.model_optim_params_list.append({'params': self.networks[key].parameters(),
                                                 'lr': optim_params['lr'],
                                                 'momentum': optim_params['momentum'],
                                                 'weight_decay': optim_params['weight_decay']})

    def init_criterions(self):

        criterion_defs = self.config['criterions']
        self.criterions = {}
        self.criterion_weights = {}

        for key, val in criterion_defs.items():
            def_file = val['def_file']
            loss_args = val['loss_params'].values()
            self.criterions[key] = source_import(def_file).create_loss(*loss_args).to(self.device)
            self.criterion_weights[key] = val['weight']
          
            if val['optim_params']:
                print('Initializing criterion optimizer.')
                optim_params = val['optim_params']
                optim_params = [{'params': self.criterions[key].parameters(),
                                'lr': optim_params['lr'],
                                'momentum': optim_params['momentum'],
                                'weight_decay': optim_params['weight_decay']}]
                # Initialize criterion optimizer and scheduler
                self.criterion_optimizer, self.criterion_optimizer_scheduler = self.init_optimizers(optim_params)
            else:
                self.criterion_optimizer = None

    def init_optimizers(self, optim_params):
        optimizer = optim.SGD(optim_params)
        scheduler = optim.lr_scheduler.StepLR(optimizer,
                                              step_size=self.scheduler_params['step_size'],
                                              gamma=self.scheduler_params['gamma'])
        return optimizer, scheduler    

        
    def train(self):

        # When training the network
        phase = "train"
        print_str = ['Phase: train']
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        # Initialize best model
        best_model_weights = {}
        best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
        best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
        best_acc = 0.0
        best_epoch = 0

        end_epoch = self.training_opt['num_epochs']
        # Loop over epochs
        for epoch in range(1, end_epoch + 1):

            for model in self.networks.values():
                model.train()
                model.cuda()
                
            torch.cuda.empty_cache()
            

            # Iterate over dataset
            epoch_loss = 0.0
            for step, batch in enumerate(self.data):

                #print("inside batch loop")
                #print(batch)

                # Break when step equal to epoch step
                # if step == self.epoch_steps:
                #     break

                inputs = batch['image']
                labels = batch['name_id']
                objectids = batch['objectid']


                inputs, labels = inputs.cuda(), labels.cuda()

                # If on training phase, enable gradients
                with torch.set_grad_enabled(True):
                        
                    # If training, forward with loss, and no top 5 accuracy calculation


                    self.features, self.feature_maps = self.networks['feat_model'](inputs)
                    centroids=self.memory['centroids']
                    feature_ext = False
                    if not feature_ext:
                        # During training, calculate centroids if needed to 
                        if phase != 'test':
                            if centroids and 'FeatureLoss' in self.criterions.keys():
                                self.centroids = self.criterions['FeatureLoss'].centroids.data
                            else:
                                self.centroids = None

                                # Calculate logits with classifier
                        self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, self.centroids)                           


                    self.loss_perf = self.criterions['PerformanceLoss'](self.logits, labels) * self.criterion_weights['PerformanceLoss']

                     # Add performance loss to total loss
                    self.loss = self.loss_perf

                     # Apply loss on features if set up
                    if 'FeatureLoss' in self.criterions.keys():
                        self.loss_feat = self.criterions['FeatureLoss'](self.features, labels)
                        self.loss_feat = self.loss_feat * self.criterion_weights['FeatureLoss']
                        self.loss += self.loss_feat

                    epoch_loss = epoch_loss + self.loss
                
                    self.model_optimizer.zero_grad()
                    if self.criterion_optimizer:
                        self.criterion_optimizer.zero_grad()
                    # Back-propagation from loss outputs
                    self.loss.backward()
                    self.model_optimizer.step()
                    if self.criterion_optimizer:
                        self.criterion_optimizer.step()


                    # Output minibatch training results
                    #if step % self.training_opt['display_step'] == 0:
                    # while(1):
                    #     minibatch_loss_feat = self.loss_feat.item() \
                    #         if 'FeatureLoss' in self.criterions.keys() else None
                    #     minibatch_loss_perf = self.loss_perf.item()
                    #     _, preds = torch.max(self.logits, 1)
                    #     minibatch_acc = mic_acc_cal(preds, labels)

                    #     print_str = ['Epoch: [%d/%d]' 
                    #                  % (epoch, self.training_opt['num_epochs']),
                    #                  'Step: %d' 
                    #                  % (step),
                    #                  'Minibatch_loss_feature: %.3f' 
                    #                  % (minibatch_loss_feat) if minibatch_loss_feat else '',
                    #                  'Minibatch_loss_performance: %.3f' 
                    #                  % (minibatch_loss_perf),
                    #                  'Minibatch_accuracy_micro: %.3f'
                    #                   % (minibatch_acc)]
                    #     print(print_str)
                    #     print_write(print_str, self.log_file)

            _, preds = torch.max(self.logits, 1)
            epoch_accuracy = (preds == labels).sum().item() / len(labels)
            print("Epoch-train-loss:'",epoch_loss.item(), " Epoch-train-accuracy:'", epoch_accuracy)
            
            
            #After every epoch, validation
            res = self.eval()

            # # Under validation, the best model need to be updated
            if res > best_acc:
                best_epoch = copy.deepcopy(epoch)
                best_acc = copy.deepcopy(res)
                best_centroids = copy.deepcopy(self.centroids)
                best_model_weights['feat_model'] = copy.deepcopy(self.networks['feat_model'].state_dict())
                best_model_weights['classifier'] = copy.deepcopy(self.networks['classifier'].state_dict())
                self.save_model(epoch, best_epoch, best_model_weights, best_acc, centroids=best_centroids)

        print('Training Complete.')      
        print('Done')

    def eval(self, phase='val', openset=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f' 
                  % self.training_opt['open_threshold'])
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()
            model.cuda()

        self.total_logits = torch.empty((0, self.training_opt['num_classes']+1)).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_paths = np.empty(0)

        # Iterate over dataset
        for valid in tqdm(self.val_data):
            inputs, labels = valid["image"].to(self.device), valid["name_id"].to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                centroids = self.memory['centroids']
                phase = 'val'
                self.features, self.feature_maps = self.networks['feat_model'](inputs)
                feature_ext=False
                # If not just extracting features, calculate logits
                if not feature_ext:

                    # During training, calculate centroids if needed to 
                    if phase != 'test':
                        if centroids and 'FeatureLoss' in self.criterions.keys():
                            self.centroids = self.criterions['FeatureLoss'].centroids.data
                        else:
                            self.centroids = None

                    # Calculate logits with classifier
                    self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, self.centroids)
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
        #        self.total_paths = np.concatenate((self.total_paths, paths))
        

        
  

        #For top1
        #probs = F.softmax(self.total_logits.detach(), dim=1)#.max(dim=1)
        #print(probs.shape)
        #print( self.total_labels.shape)

        #precision = dict()
        #recall = dict()
        #average_precision = dict()

        #n_classes = 201
        #for i in range(n_classes):
        #    precision[i], recall[i], _ = metrics.precision_recall_cruve(

   
        #Uncomment for topk
        #probs, preds = F.softmax(self.total_logits.detach(), dim=1).topk(k=5,dim=1)#.max(dim=1)

        batch_size = self.total_labels.size(0)

        _, pred = self.total_logits.topk(5, 1, True, True)
        pred = pred.t()
        correct = pred.eq(self.total_labels.view(1, -1).expand_as(pred))

        #print(correct.shape)

        correct_k = correct[:5].reshape(-1).float().sum(0)
        res = (correct_k.mul_(100.0 / batch_size)).item()
        print("Eval-Accuracy :", res)
        return res
    
            
    def centroids_cal(self, data):

        centroids = torch.zeros(self.training_opt['num_classes']+1,
                                   self.training_opt['feature_dim']).cuda()

        print('Calculating centroids.')

        total_classes = self.training_opt['num_classes'] + 1

        for model in self.networks.values():
            model.eval()

        with torch.set_grad_enabled(False):
            
            for p in tqdm(data):
                inputs, labels = p["image"].to(self.device), p["name_id"].to(self.device)

                # Calculate Features of each training data
                self.features, self.feature_maps = self.networks['feat_model'](inputs)

                feature_ext=True
                # If not just extracting features, calculate logits
                if not feature_ext:

                    # During training, calculate centroids if needed to 
                    if phase != 'test':
                        if centroids and 'FeatureLoss' in self.criterions.keys():
                            self.centroids = self.criterions['FeatureLoss'].centroids.data
                        else:
                            self.centroids = None

                    # Calculate logits with classifier
                    self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, self.centroids)
        
                # Add all calculated features to center tensor
                for i in range(len(labels)):
                    label = labels[i]
                    centroids[label] += self.features[i]

        # Average summed features with class count
        centroids /= torch.tensor(class_count(total_classes, data)).float().unsqueeze(1).cuda()

        return centroids

    def load_model(self):
            
        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')
        
        print('Validation on the best model.')
        print('Loading model from %s' % (model_dir))
        
        checkpoint = torch.load(model_dir)          
        model_state = checkpoint['state_dict_best']
        
        self.centroids = checkpoint['centroids'] if 'centroids' in checkpoint else None
        
        for key, model in self.networks.items():

            weights = model_state[key]
            weights = {k: weights[k] for k in weights if k in model.state_dict()}
            # model.load_state_dict(model_state[key])
            model.load_state_dict(weights)
        
    def save_model(self, epoch, best_epoch, best_model_weights, best_acc, centroids=None):
        
        model_states = {'epoch': epoch,
                'best_epoch': best_epoch,
                'state_dict_best': best_model_weights,
                'best_acc': best_acc,
                'centroids': centroids}

        model_dir = os.path.join(self.training_opt['log_dir'], 
                                 'final_model_checkpoint.pth')

        torch.save(model_states, model_dir)
            
    def output_logits(self, openset=False):
        filename = os.path.join(self.training_opt['log_dir'], 
                                'logits_%s'%('open' if openset else 'close'))
        print("Saving total logits to: %s.npz" % filename)
        np.savez(filename, 
                 logits=self.total_logits.detach().cpu().numpy(), 
                 labels=self.total_labels.detach().cpu().numpy(),
                 paths=self.total_paths)

    def infer(self, phase='test', openset=False):

        print_str = ['Phase: %s' % (phase)]
        print_write(print_str, self.log_file)
        time.sleep(0.25)

        if openset:
            print('Under openset test mode. Open threshold is %.1f' 
                  % self.training_opt['open_threshold'])
 
        torch.cuda.empty_cache()

        # In validation or testing mode, set model to eval() and initialize running loss/correct
        for model in self.networks.values():
            model.eval()
            model.cuda()

        self.total_logits = torch.empty((0, self.training_opt['num_classes'])).to(self.device)
        self.total_labels = torch.empty(0, dtype=torch.long).to(self.device)
        self.total_paths = np.empty(0)

        # Iterate over dataset
        for valid in tqdm(self.val_data):
            inputs= valid["image"]. to(self.device)

            # If on training phase, enable gradients
            with torch.set_grad_enabled(False):

                # In validation or testing
                centroids = self.memory['centroids']
                self.features, self.feature_maps = self.networks['feat_model'](inputs)
                feature_ext=False
                # If not just extracting features, calculate logits
                if not feature_ext:

                    # During training, calculate centroids if needed to 
                    if phase != 'test':
                        if centroids and 'FeatureLoss' in self.criterions.keys():
                            self.centroids = self.criterions['FeatureLoss'].centroids.data
                        else:
                            self.centroids = None

                    # Calculate logits with classifier
                    self.logits, self.direct_memory_feature = self.networks['classifier'](self.features, self.centroids)
                self.total_logits = torch.cat((self.total_logits, self.logits))
                self.total_labels = torch.cat((self.total_labels, labels))
        #        self.total_paths = np.concatenate((self.total_paths, paths))
        

        
  

        #For top1
        #probs = F.softmax(self.total_logits.detach(), dim=1)#.max(dim=1)
        #print(probs.shape)
        #print( self.total_labels.shape)

        #precision = dict()
        #recall = dict()
        #average_precision = dict()

        #n_classes = 201
        #for i in range(n_classes):
        #    precision[i], recall[i], _ = metrics.precision_recall_cruve(

   
        #Uncomment for topk
        probs, preds = F.softmax(self.total_logits.detach(), dim=1).topk(k=3,dim=1)#.max(dim=1)

        #For printing inference results
        for i in range(len(probs)):
            print("Path: ", self.total_paths[i], preds[i], probs[i])

     
       
        ##x = metrics.functional.classification.multiclass_precision_recall_curve(probs, self.total_labels)
        ##tup = np.shape(x)
        #print(x[0])
        #print(x[1])
        #print(x[2])

        #confusion matrix
        ## probs, preds = F.softmax(self.total_logits.detach(), dim=1).max(dim=1)
        ## cm = skmetrics.confusion_matrix(self.total_labels.cpu(), preds.cpu())
        ## cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        ## df_cm = pd.DataFrame(cm, range(200), range(200))
        ## sn.set(font_scale=1.0) # for label size
        ## cmap = sn.cm.rocket_r
        ## sn.heatmap(df_cm, cmap=cmap) # font size
        ## plt.savefig("cm1.png")

        #PR curve
        # For each class
        ##precision = dict()
        ##recall = dict()
        ##average_precision = dict()
        ##self.total_logits=self.total_logits.cpu()
        ##probs = F.softmax(self.total_logits.detach(), dim=1)
        ##labels = F.one_hot(self.total_labels.cpu())
        #for i in range(200):
        #    precision[i], recall[i], _ =  skmetrics.precision_recall_curve(y_true=labels[i], probas_pred=probs[i])
                                                        
        #    average_precision[i] =  skmetrics.average_precision_score(y_true=labels[i], y_score=probs[i])
        # A "micro-average": quantifying score on all classes jointly
        # precision["weighted"], recall["weighted"], _ = skmetrics.precision_recall_curve(y_true=labels.numpy().ravel(),probas_pred=probs.numpy().ravel())
        # average_precision["weighted"] = skmetrics.average_precision_score(y_true=labels, y_score=probs, average="weighted")
        # print('Average precision score, weight-averaged over all classes: {0:0.2f}'.format(average_precision["weighted"]))
        # plt.figure()
        # plt.step(recall['weighted'], precision['weighted'], where='post')
        # plt.xlabel('Recall')
        # plt.ylabel('Precision')
        # plt.ylim([0.0, 1.05])
        # plt.xlim([0.0, 1.0])
        # plt.plot( [0.0,1.0],[1.0,0.0] )
        # plt.title('Average precision score, micro-averaged over all classes: AP={0:0.2f}'.format(average_precision["weighted"]))
        # plt.savefig("pr_curve_weighted.png")
        
     

        
        
        # if openset:
        #     preds[probs < self.training_opt['open_threshold']] = -1
        #     self.openset_acc = mic_acc_cal(preds[self.total_labels == -1],
        #                                     self.total_labels[self.total_labels == -1])
        #     print('\n\nOpenset Accuracy: %.3f' % self.openset_acc)

        # Calculate the overall accuracy and F measurement
        #self.eval_acc_mic_top1= mic_acc_cal(preds[self.total_labels != -1],
        #                                    self.total_labels[self.total_labels != -1])
        #self.eval_f_measure = F_measure(preds, self.total_labels, openset=openset,
        #                                theta=self.training_opt['open_threshold'])
        #self.many_acc_top1, \
        #self.median_acc_top1, \
        #self.low_acc_top1 = shot_acc(preds[self.total_labels != -1],
        #                             self.total_labels[self.total_labels != -1], 
        #                             self.data['train'])
        # Top-1 accuracy and additional string
