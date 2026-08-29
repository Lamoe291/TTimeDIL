import logging
import numpy as np
import torch
import time
from torch import nn
from torch.serialization import load
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import IncrementalNet
import utils.inc_net as inc_net
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from sklearn.metrics import average_precision_score
import copy

from collections import defaultdict
import pandas as pd 

import timm
from backbone.lora import LoRA_ViT_timm
from backbone.lora_efficient import LoRA_ViT_timm as LoRA_ViT_timm_efficient
from backbone.linears import SimpleLinear
import torch.distributed as dist
import random

import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)
        self.class_to_taskID_mapping = {}
        self.task_prototypes = {}
        self.task_precisions = {}
        self.task_var = {}
        self.task_R = {}
        self.task_prec_diag = {}
        self.task_P =  {}
        self.shared_U = None
        self.task_eigenvalues = {}
        self.cov_sum = torch.zeros(768,768).to(self._device) #None
        self.shared_cov = None
        self.task_effective_dim = {}
        self.task_prototypes_residuals = {}
        self.all_class_means = {}
        self.class_prototypes = {}
        self.tt_stats = {}
        #self.use_tt_merging = True
        self.use_task_prototypes = True
        self.random_projection = False#True
        self.tukey = False
        #self.task_specific_prototypes = False
        self.weighted_head = False #True
        self.normalization = False
        self.train_merge = False
        #self.use_effective_dim = False
        #self.task_separation_regularization = 0.0001
        self.random_matrix  = torch.randn(2000, 768).to(self._device) / (768 ** 0.5)
        self.domain_heads_weights = torch.Tensor([]).to(self._device)
        self.domain_heads_bias = torch.Tensor([]).to(self._device)
        #self.domain_order = args.


    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        for i in range(self._known_classes, self._total_classes):
            self.class_to_taskID_mapping[i] = self._cur_task

        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
        )
        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        
        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers# batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[self._device], output_device=self._device, find_unused_parameters=True)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

        self._train(self.train_loader, self.test_loader)

        # to test
        # self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def incremental_train_dil(self, data_manager):
        self._cur_task += 1
        self._total_classes = data_manager.nb_classes#19
        self.domain_order = data_manager._class_order
        print('num_classes', self._total_classes)
        
        if self._cur_task == 0:
            self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        #print(data_manager._class_order[self._cur_task])
        if self.args['dataset']!='bigearthnet' and self.args['dataset']!='core50':
            train_dataset = data_manager.get_dataset_ml(
                    domains=[data_manager._class_order[self._cur_task]],
                    source="train",
                    mode="train",
                )
            if self.args.get('validation',False):
                test_dataset = data_manager.get_dataset_ml(
                    domains=data_manager._class_order[:self._cur_task+1],
                    source="validation",
                    mode="test",
                    return_domains=True,
                )
            else:    
                test_dataset = data_manager.get_dataset_ml(
                    data_manager._class_order[:self._cur_task+1], source="test", mode="test", return_domains=True,
                )
            self.test_dataset = test_dataset
            print('train samples', len(train_dataset))
            # self.train_loader = DataLoader(
            #     train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=2
            # )
            
            print('test samples', len(test_dataset))

        elif self.args['dataset']=='core50':
            train_dataset = data_manager.get_dataset_ml(
                    domains=[data_manager._class_order[self._cur_task]],
                    source="train",
                    mode="train",
                )
            if self.args.get('validation',False):
                test_dataset = data_manager.get_dataset_ml(
                    domains=data_manager._class_order[:self._cur_task+1],
                    source="validation",
                    mode="test",
                    return_domains=True,
                )
            else:    
                test_dataset = data_manager.get_dataset_ml(
                    domains=np.array(['s3','s7','s10']), source="test", mode="test", return_domains=True,
                )
            self.test_dataset = test_dataset
            print('train samples', len(train_dataset))
            # self.train_loader = DataLoader(
            #     train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=2
            # )
            
            print('test samples', len(test_dataset))

        else:
            train_dataset = data_manager.get_dataset_ben(
                domains=[data_manager._class_order[self._cur_task]],
                source="train",
                mode="train",
            )
            print('train samples', len(train_dataset))
            
            if self.args.get('validation',False):
                test_dataset = data_manager.get_dataset_ben(
                    data_manager._class_order[:self._cur_task+1], source="validation", mode="test"
                )
            else:    
                test_dataset = data_manager.get_dataset_ben(
                    data_manager._class_order[:self._cur_task+1], source="test", mode="test"
                )
            print('test samples', len(test_dataset))
            self.test_dataset = test_dataset

        self.train_loader = DataLoader(
                train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
            )
        
        if self.args.get('efficient_inference',False):
            test_bz = self.args["batch_size"]
        else:
            test_bz = 1
        self.test_loader = DataLoader(
            test_dataset, batch_size=test_bz, shuffle=False, num_workers=num_workers
        )

        if len(self._multiple_gpus) > 1:
            self._network = nn.DataParallel(self._network, self._multiple_gpus)
            # model = nn.parallel.DistributedDataParallel(model, device_ids=[self._device], output_device=self._device, find_unused_parameters=True)
        # if len(self._multiple_gpus) > 1:
        #     self._network = self._network.module

        self._train(self.train_loader, self.test_loader)

        # to test
        # self._network.to(self._device)
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module
            

    def update_network(self, index=True):
        backbone_type = self.args.get("backbone_type", "vit_base_patch16_224")
        print(f"[update_network] selecting backbone '{backbone_type}' from args")
        model = inc_net._create_pretrained_vit_model(backbone_type, pretrained=True, num_classes=0)[0]

        # SD-LoRA-RR
        '''
        if self._cur_task >=4 and self._cur_task <8:
            rank = 8 #8
        elif self._cur_task >=8:
            rank = 6 #6
        # elif self._cur_task >=8:
        #     rank = 4
        else:
            rank = 10
        '''
        rank = self.args.get('lora_rank',10)
        if self.args.get('efficient_inference',False):
            model = LoRA_ViT_timm_efficient(vit_model=model.eval(), r=rank, num_classes=10, index=index, increment= self.args['increment'], filepath=self.args['filepath'] + self.args['prefix'] + '/' , 
            cur_task_index= self._cur_task, top_k=self.args.get('top_k',3))
        else:
            model = LoRA_ViT_timm(vit_model=model.eval(), r=rank, num_classes=10, index=index, increment= self.args['increment'], filepath=self.args['filepath'] + self.args['prefix'] + '/' , 
            cur_task_index= self._cur_task)
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        
        self._network.to(self._device)
        self._network.backbone.deactivate_eval()
        if self._cur_task == 0:
            optimizer = self.get_optimizer(lr=self.args['init_lr'], head_lr=self.args.get('init_head_lr', None))
            scheduler = self.get_scheduler(optimizer)

            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            self._network.backbone = self.update_network(index=False)
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)       
            self._network.to(self._device) 

            optimizer = self.get_optimizer(lr=self.args['lrate'], head_lr=self.args.get('head_lr', None))
            scheduler = self.get_scheduler(optimizer)

            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_lora_name = self.args['filepath']  + self.args['prefix'] + '/'

        if len(self._multiple_gpus) > 1:
            self._network.module.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.module.save_fc(save_lora_name, self._cur_task)
        else:
            self._network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.save_fc(save_lora_name, self._cur_task)
            #head = SimpleLinear(in_features=768, out_features=19)
            #temp_weights = torch.load(self.args['filepath'] + self.args['prefix'] + '/CLs_weight'+str(i)+'.pt') 
            #temp_bias = torch.load(self.args['filepath'] + self.args['prefix'] + '/CLs_bias'+str(i)+'.pt') 
            #head.weight.data = temp_weights.data.cuda()
            #head.bias.data = temp_bias.data.cuda()
            self.domain_heads_weights = torch.cat([self.domain_heads_weights,copy.deepcopy(self._network.fc.weight.data)])
            self.domain_heads_bias = torch.cat([self.domain_heads_bias,copy.deepcopy(self._network.fc.bias.data)])

    
    def get_optimizer(self, lr=None, head_lr=None):

        # Base learning rate
        optim_lr = lr if lr is not None else self.args['lrate']

        # Classification-head learning rate
        if head_lr is None:
            head_lr = optim_lr

        # Separate parameters into two groups
        head_params = []
        base_params = []

        for name, param in self._network.named_parameters():
            if not param.requires_grad:
                continue

            if 'fc' in name:
                head_params.append(param)
                #logging.info(f"Parameter: {name}, shape: {param.shape}, requires_grad: {param.requires_grad} (head)")
            else:
                base_params.append(param)
                #logging.info(f"Parameter: {name}, shape: {param.shape}, requires_grad: {param.requires_grad} (base)")


        param_groups = [
            {
                'params': base_params,
                'lr': optim_lr
            },
            {
                'params': head_params,
                'lr': head_lr
            }
        ]
        logging.info(f"Base learning rate: {optim_lr}, Head learning rate: {head_lr}")

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                param_groups,
                momentum=0.9,
                weight_decay=self.weight_decay
            )

        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                param_groups,
                betas=(0.9, 0.999)
            )

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'], eta_min=self.args['min_lr'])
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler
    
    def get_head_mask_for_task_id(self, task_id, num_classes):
        mask = torch.zeros(num_classes)
        mask = mask > 0.0
        classes_per_task = int(num_classes/(self._cur_task+1))
        #print(classes_per_task)
        task_id_based_class_ids = [*range(task_id*classes_per_task,(task_id+1)*classes_per_task)]
        #print(task_id_based_class_ids)
        mask[task_id_based_class_ids] = 1
        #print(mask)
        return mask

    def _eval_cnn(self, loader):
        self._network.eval()
        self._network.backbone.activate_eval()
        top_k = min(self.args.get("top_k",self._cur_task+1), self._cur_task+1)

        all_prototypes, all_precisions = None, None
        if not self.args.get('class_based_prototypes', False):
            torch.cuda.synchronize()
            tstacking_start = time.time()
            all_prototypes = torch.stack([self.task_prototypes[i] for i in range(self._cur_task+1)], dim=0)
            all_precisions = torch.stack([self.task_precisions[i] for i in range(self._cur_task+1)], dim=0)
            tstacking_time = time.time() - tstacking_start
            logging.info('stacking domain stats: {}'.format(tstacking_time))

        if self.args.get('weighting_stats',True):
            weighting_stats = WeightingStats(args=self.args, num_tasks=10, current_task=self._cur_task, max_k=5)
        y_pred, y_true = [], []

        if self.args.get('efficient_inference'):
            large_head_weights = self.domain_heads_weights #torch.cat([head.weight for head in self.domain_heads])  # shape: (N, ...)
            large_head_bias = self.domain_heads_bias # torch.cat([head.bias for head in self.domain_heads])

        if self.args.get('effective_dim',False):
            task_effective_dim = torch.Tensor([self.task_effective_dim[i] for i in range(self._cur_task+1)])
            print('task effective dim: ', task_effective_dim)

        if self.args.get('task_specific_prototypes',False):
            task_residuals = torch.stack([self.task_prototypes_residuals[j] for j in range(self._cur_task+1)],dim=0)

        elapsed = 0
        elapsed_tt_weights_computation = 0
        task_weights_ranked = torch.Tensor([])
        task_weights_heat_map = defaultdict(list)
        if self.args['dataset'] == 'bigearthnet':
            df = pd.read_parquet(self.args["data_dir"] + "/BigEarthNet-V2/metadata.parquet")
            lookup = dict(zip(df["patch_id"], df["country"]))
        for i, batch in enumerate(loader):
            if self.args['dataset'] != 'bigearthnet':
                (image_id, inputs, targets, domains) = batch[0],batch[1],batch[2], batch[3]
            else:
                (inputs, targets, image_id) = batch[0],batch[1],batch[2]
                domains = [lookup[p] for p in image_id]
            #if i > 2:
            #    break
            inputs = inputs.to(self._device)
            batch_size = inputs.shape[0]
            #print(domains)
            with torch.no_grad():
                # outputs = self._network.forward(inputs, eval=True)['logits']
                # print('outputs', outputs['logits'])
                #print('targets', targets)
                if not self.args.get('domain_incremental_learning',False):
                    sample_task_id = self.class_to_taskID_mapping[targets.item()]
                else:
                    sample_task_id = 0 #self.class_to_taskID_mapping[0]
                #print(sample_task_id)
                torch.cuda.synchronize()
                t0 = time.time()
                if self.use_task_prototypes:
                    
                    if self.args.get('task_specific_prototypes',False):
                        #task_specific_features = []

                        helper_weights = torch.zeros(self._cur_task+1).to(self._device)
                        #helper_weights[0] = 1
                        self._network.backbone.set_internal_tt_weights(helper_weights)
                        features = self._network.forward(inputs)['features']#['logits']
                        
                        if self.random_projection:
                                print('use random projection for pt')
                                features = self._random_project(features, nonlinear='relu')
                        if self.tukey:
                            print('use tukey power transform')
                            features = torch.pow(features,exponent=0.5)


                        task_specific_features = (features[:, None, :] + task_residuals[None, :, :]).reshape(-1, 768)
                        #print(task_specific_features.shape)
                        distances = self.compute_distance_to_task_specific_prototypes(all_features=task_specific_features, distance_metric=self.args.get('distance','mahalanobis'))
                        #print(distances.shape)

                    else:
                        if self.args.get('efficient_inference',False):
                            helper_weights = torch.zeros(batch_size, self._cur_task+1).to(self._device)
                        else:
                            helper_weights = torch.zeros(self._cur_task+1).to(self._device)
                        #helper_weights[0] = 1
                        self._network.backbone.set_internal_tt_weights(helper_weights)
                        features = self._network.forward(inputs)['features']#['logits']
                        #print('feature shape', features.shape)

                    #self._network.backbone.set_internal_tt_weights(helper_weights)
                    #features = self._network.forward(inputs)['features']#['logits']
                        if self.tukey:
                                print('use tukey power transform')
                                features = torch.pow(features,exponent=0.5)
                        if self.random_projection:
                            print('use random projection for pt')
                            features = self._random_project(features, nonlinear='relu')
                        #print('features shape', features.shape)
                        
                        
                        raw_distances = self.compute_distance_to_prototypes(features=features, prototypes=all_prototypes, precisions=all_precisions, distance_metric=self.args.get('distance','mahalanobis'))

                        if self.args.get('class_based_prototypes', False):
                            class_agg = self.args.get('class_to_task_agg', 'min')
                            task_weights, distances = self.aggregate_class_distances_to_task_weights(
                                class_distance_list=raw_distances,
                                class_agg_strategy=class_agg,
                                weighting_strat=self.args.get('task_weighting_strat', 'softmax'),
                            )
                        else:
                            distances = raw_distances
                        
                        if self.args.get('effective_dim',False):#use_effective_dim:
                            distances = distances /task_effective_dim.to(self._device)
                        #distances = get_task_distances(features=features, class_prototypes=self.class_prototypes)
                        
                    #print(distances)
                    #distances = torch.ones(self._cur_task+1).to(self._device)
                    #distances[sample_task_id] = 0

                    if not self.args.get('class_based_prototypes', False):
                        if self.args.get('distance','mahalanobis')=='gaussian_likelihood' and self.args.get('diki_analog',True):
                            task_weights = self.get_task_weights(distances=distances,weighting_strat='max')#
                        else:
                            if self.args.get('distance','mahalanobis')=='gaussian_likelihood':
                                task_weights = distances / distances.sum(dim=1, keepdim=True)#self.get_task_weights(distances=distances,weighting_strat='softmax')#
                            else:task_weights = self.get_task_weights(distances=distances,weighting_strat=self.args.get('task_weighting_strat', 'softmax'))#
                    
                    if not self.args.get('multi-label',False):
                        predicted_task_id = torch.argmax(task_weights).item()
                    else:
                        predicted_task_id = 0

                    #print(task_weights)
                    #print('task weights shape', task_weights.shape)

                    #smoothed_weights = self.sim_matrix @ task_weights
                    #task_weights = smoothed_weights / smoothed_weights.sum()
                    #task_weights = torch.ones_like(task_weights)
                    #task_weights = task_weights/task_weights.norm()
                    if self.args.get('weighting_stats',True):
                        weighting_stats.update(task_weights.detach().cpu(), correct_task=sample_task_id)
                else:
                    task_weights = torch.ones(self._cur_task+1) / (self._cur_task+1)
                    if self.args.get('weighting_stats',True):
                        weighting_stats.update(task_weights.detach().cpu(), correct_task=sample_task_id)
                
                task_weights = task_weights.to(self._device)

                torch.cuda.synchronize()
                elapsed_tt_weights_computation += time.time() - t0
                #print(task_weights.shape)
                if self.args.get('get_weighting_stats', False):
                    sorted_task_weights, _ = torch.sort(task_weights, descending=True)
                    task_weights_ranked = torch.cat((task_weights_ranked, sorted_task_weights.cpu()), dim=0)

                    for sample_weights, sample_domain in zip(task_weights, domains):
                        task_weights_heat_map[sample_domain].append(sample_weights.cpu())
                if not self.args.get('domain_incremental_learning',True):
                    self._network.backbone.set_internal_tt_weights(task_weights)
                    outputs = self._network.forward(inputs)['logits']
                    #self._network.backbone.reset_internal_tt_weights(bz=batch_size,full_weight_task_id=self._cur_task)

                else:
                    #if not self.args.get('merged_lora',True):
                    if self.args.get('merging', 'all')!='all' and self.args.get('merging', 'all')!='lora':
                        helper_weights = torch.zeros_like(task_weights).to(self._device)
                        helper_weights[:,self._cur_task] = 1
                        self._network.backbone.set_internal_tt_weights(helper_weights)
                    else:
                        self._network.backbone.set_internal_tt_weights(task_weights)
                    
                    #weights = copy.deepcopy(self._network.fc.weights.data) * task_weights[-1]
                    if self._cur_task > 0:
                        outputs =  self._network.forward(inputs)['features']
                        #self._network.backbone.reset_internal_tt_weights(bz=batch_size,full_weight_task_id=self._cur_task)
                        #print(len(self._network.heads_w))
                        if self.args.get('efficient_inference',False):
                            
                            
                            #weights = torch.cat([head.weight for head in class_heads])  # shape: (N, ...)
                            #bias = torch.cat([head.bias for head in class_heads])  # shape: (N, ...)
                            #result_w = (coeffs_w * weights).sum(dim=0)
                            #result_b = (coeffs_b * bias).sum(dim=0)                         
                            
                            outputs = F.linear(outputs, large_head_weights, large_head_bias)
                            #print(outputs.shape)

                            #if self.args.get('merged_head',True) or 
                            if self.args.get('merging', 'all')=='all' or self.args.get('merging', 'all')=='head':
                                #B = batch_size
                                #T = self._cur_task+1
                                #C = self.args["nb_classes"]
                                #outputs = (outputs.view(B, T, C) * task_weights.unsqueeze(-1)).sum(dim=1)

                                #outputs = task_weights * outputs # .view(len(class_heads), *([1] * (weights.dim() - 1)))  
                                #outputs = task_weights * outputs # .view(len(class_heads), *([1] * (bias.dim() - 1))) 
                                # 
                                B = batch_size
                                T = self._cur_task + 1
                                C = self.args["nb_classes"]

                                weights = task_weights[:, :T]                        # (B, T)


                                top_k = min(self.args.get("top_k",T), T)
                                topk_weights, topk_indices = weights.topk(top_k, dim=1)  # (B, k)
                                topk_weights = topk_weights / topk_weights.sum(dim=1, keepdim=True)

                                # gather only top-k class logits per sample
                                outputs_3d = outputs.view(B, T, C)
                                idx = topk_indices.unsqueeze(-1).expand(B, top_k, C)     # (B, k, C)
                                outputs_topk = outputs_3d.gather(1, idx)                  # (B, k, C)

                                outputs = (outputs_topk * topk_weights.unsqueeze(-1)).sum(dim=1)  # (B, C) 

                        else:
                            #if self.args.get('merged_head',True):
                            if self.args.get('merging', 'all')=='all' or self.args.get('merging', 'all')=='head':
                                weights = torch.stack([head.weight for head in class_heads])  # shape: (N, ...)
                                bias = torch.stack([head.bias for head in class_heads])  # shape: (N, ...)
                                coeffs_w = task_weights.view(len(class_heads), *([1] * (weights.dim() - 1)))  
                                coeffs_b = task_weights.view(len(class_heads), *([1] * (bias.dim() - 1)))  
                                result_w = (coeffs_w * weights).sum(dim=0)
                                result_b = (coeffs_b * bias).sum(dim=0)
                            else:
                                weights = torch.cat([head.weight for head in class_heads])  # shape: (N, ...)
                                bias = torch.cat([head.bias for head in class_heads])  # shape: (N, ...)
                                result_w = weights
                                result_b = bias
                            #result_w = result_w + task_weights[-1]* self._network.fc.weight.data
                            #result_b = result_b + task_weights[-1]* self._network.fc.bias.data
                            
                            outputs = F.linear(outputs, result_w, result_b)
                    else:
                        outputs =  self._network.forward(inputs)['logits']
                        #self._network.backbone.reset_internal_tt_weights(bz=batch_size,full_weight_task_id=self._cur_task) 

                    #if not self.args.get('merged_head', True):
                    if self.args.get('merging', 'all')!='all' and self.args.get('merging', 'all')!='head':
                        B, CT = outputs.shape
                        C = self.args["nb_classes"]#self._total_classes - self._known_classes
                        T = CT // C
                        #print('no merged')
                        outputs = outputs.view(B, T, C)      # [B, T, C]
                        #outputs = outputs.mean(dim=1)#.values
                        outputs = outputs.max(dim=1).values

                torch.cuda.synchronize()
                elapsed += time.time() - t0

            if not self.args.get('multi-label',False):
                #print(outputs.shape)
                predicts = torch.topk(outputs, k=self.topk, dim=1, largest=True, sorted=True)[1]  # [bs, topk]
            else:
                predicts = torch.sigmoid(outputs) #> 0.5
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

            if not self.use_task_prototypes:
                ## to ensure running
                predicted_task_id = 0
                distances = torch.ones(self._cur_task+1).to(self._device)


            if self.args.get('weighting_stats',True):
                if not self.args.get('multi-label',False):
                    weighting_stats.update_confusion(task_correct=(predicted_task_id==sample_task_id), class_correct=(predicts[0][0].cpu()==targets[0].cpu()))
                else:
                    weighting_stats.update_confusion(task_correct=(predicted_task_id==sample_task_id), class_correct=True)
                if (predicted_task_id==sample_task_id):
                    sorted_dist,_ = torch.sort(distances)
                    sorted_t_weights, _ = torch.sort(task_weights)
                    if len(sorted_dist)>1:
                        dist_difference_1_2 = sorted_dist[1]-sorted_dist[0]
                        weight_difference_1_2 = sorted_t_weights[1]-sorted_t_weights[0]
                        weighting_stats.update_additional_stats(key='task_dist_diff_1_2',value=dist_difference_1_2)
                        weighting_stats.update_additional_stats(key='task_weight_diff_1_2',value=weight_difference_1_2)
                if (predicted_task_id!=sample_task_id):
                    sorted_dist,_ = torch.sort(distances)
                    sorted_t_weights, _ = torch.sort(task_weights)
                    if len(sorted_dist)>1:
                        dist_difference_1_2 = sorted_dist[1]-sorted_dist[0]
                        weight_difference_1_2 = sorted_t_weights[1]-sorted_t_weights[0]
                        weighting_stats.update_additional_stats(key='no_task_dist_diff_1_2',value=dist_difference_1_2)
                        weighting_stats.update_additional_stats(key='no_task_weight_diff_1_2',value=weight_difference_1_2)
            

                
            # print('y_pred', np.concatenate(y_pred))
            # print('y_true', y_true)
        logging.info('Compute tt weights time: {}'.format(elapsed_tt_weights_computation))
        self.total_testing_time += elapsed
        if self.args.get('get_weighting_stats', False):
            task_weights_ranked_means = task_weights_ranked.mean(dim=0)
            task_weights_ranked_stds = task_weights_ranked.std(dim=0)


            if self.args['dataset']=='bigearthnet':
                task_weights_heat_map = {k: torch.stack(task_weights_heat_map[k]).mean(dim=0) for k in self.domain_order[:self._cur_task+1]}
            else:
                task_weights_heat_map = {k: torch.stack(v).mean(dim=0) for k, v in task_weights_heat_map.items()}
            logging.info('Task weights ranked means: {}'.format(task_weights_ranked_means))
            logging.info('Task weights ranked stds: {}'.format(task_weights_ranked_stds))
            logging.info('Heat map means: {}'.format(task_weights_heat_map))
        if self.args.get('weighting_stats',True):
            results = weighting_stats.compute()
            self.tt_stats[self._cur_task] = results

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]
    
    def _eval_cnn_ML(self, loader):
        return self._eval_cnn(loader)

    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        if not self.args.get('multi-label',False):
            prog_bar = tqdm(range(self.args["init_epoch"]))
            for _, epoch in enumerate(prog_bar):
                self._network.train()
                

                losses = 0.0
                correct, total = 0, 0
                elapsed = 0
                for i, batch in enumerate(train_loader):
                    if self.args['dataset'] != 'bigearthnet':
                        (_, inputs, targets) = batch[0],batch[1],batch[2]
                    else:
                        (inputs, targets, _) = batch[0],batch[1],batch[2]

                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    logits = self._network(inputs)["logits"]
                    loss = F.cross_entropy(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    elapsed += time.time() - t0
                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)
                self.total_training_time += elapsed
                if scheduler != None:
                    scheduler.step()
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

                if epoch % 5 == 0:
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.args["init_epoch"],
                        losses / len(train_loader),
                        train_acc,
                        test_acc,
                    )
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.args["init_epoch"],
                        losses / len(train_loader),
                        train_acc,
                    )

                prog_bar.set_description(info)

                logging.info(info)

        else:
            prog_bar = tqdm(range(self.args["init_epoch"]))
            for _, epoch in enumerate(prog_bar):
                self._network.train()
                losses = 0.0
                correct, total = 0, 0
                all_targets = []
                all_preds = []
                elapsed = 0
                for i, batch in enumerate(train_loader):
                    if self.args['dataset'] != 'bigearthnet':
                        (_, inputs, targets) = batch[0],batch[1],batch[2]
                    else:
                        (inputs, targets, _) = batch[0],batch[1],batch[2]

                    inputs, targets = inputs.to(self._device), targets.to(self._device).to(torch.float32)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    logits = self._network(inputs)["logits"]
                    #targets = targets
                    #print(logits.dtype)
                    #print(targets.dtype)

                    #logits_pred = logits.sigmoid()
                    loss = F.binary_cross_entropy_with_logits(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    elapsed += time.time() - t0
                    losses += loss.item()

                    with torch.no_grad():
                        preds = torch.sigmoid(logits.detach()) #> 0.5
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())

                    #self.val_preds.append(preds.cpu())
                    #self.val_targets.append(y.cpu())
                    #ap_macro = average_precision_score(targets.cpu(),preds.cpu(),average='macro')
                    #ap_micro = average_precision_score(targets.cpu(),preds.cpu(),average='micro')
                    
                    #_, preds = torch.max(logits, dim=1)
                    #correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    #total += len(targets)
                if scheduler != None:
                    scheduler.step()
                #train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

                self.total_training_time += elapsed
                preds = torch.cat([o for o in all_preds], dim=0)
                targets = torch.cat([o for o in all_targets], dim=0)
                ap_macro = average_precision_score(targets,preds,average='macro')
                ap_micro = average_precision_score(targets,preds,average='micro')

                if epoch % 5 == 0:
                    test_ap_micro, test_ap_macro = self._compute_mAP(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f},  Train_mAP macro/micro {:.2f}/{:.2f}, Test_mAP macro/micro {:.2f}/{:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.args["init_epoch"],
                        losses / len(train_loader),
                        ap_macro,
                        ap_micro,
                        test_ap_macro,
                        test_ap_micro,
                    )
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_mAP macro {:.2f}, micro {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.args["init_epoch"],
                        losses / len(train_loader),
                        ap_macro,
                        ap_micro,
                    )

                prog_bar.set_description(info)

                logging.info(info)

    def _update_representation(self, train_loader, test_loader, optimizer, scheduler):
        
        if not self.args.get('multi-label',False):
            prog_bar = tqdm(range(self.args["epochs"]))
            for _, epoch in enumerate(prog_bar):
                self._network.train()
                losses = 0.0
                correct, total = 0, 0
                elapsed = 0
                for i, (_, inputs, targets) in enumerate(train_loader):
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    # logits = self._network(inputs)["logits"]
                    torch.cuda.synchronize()
                    t0 = time.time()
                    logits = self._network(inputs)
                    #features = logits['features']
                    logits = logits['logits'] 
                    
                    loss_clf = F.cross_entropy(logits, targets)
                    #fake_targets = targets - self._known_classes
                    #loss_clf = F.cross_entropy(
                    #    logits[:, self._known_classes :], fake_targets
                    #)
                    # print('@@@@@@@@@@@@@@loss2', loss_clf, torch.mean(ortho_loss))

                    # loss = loss_clf + 10* torch.mean(ortho_loss)

                    loss = loss_clf

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    elapsed += time.time() - t0
                    losses += loss.item()

                    _, preds = torch.max(logits, dim=1)
                    correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    total += len(targets)

                if scheduler != None:
                    scheduler.step()
                self.total_training_time += elapsed
                train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                if epoch % 5 == 0:
                    test_acc = self._compute_accuracy(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}, Test_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.args["epochs"],
                        losses / len(train_loader),
                        train_acc,
                        test_acc,
                    )
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.args["epochs"],
                        losses / len(train_loader),
                        train_acc,
                    )
                prog_bar.set_description(info)
                logging.info(info)

        else:
            prog_bar = tqdm(range(self.args["epochs"]))
            for _, epoch in enumerate(prog_bar):
                self._network.train()
                losses = 0.0
                correct, total = 0, 0
                all_targets = []
                all_preds = []
                elapsed = 0
                for i, batch in enumerate(train_loader):
                    if self.args['dataset'] != 'bigearthnet':
                        (_, inputs, targets) = batch[0],batch[1],batch[2]
                    else:
                        (inputs, targets, _) = batch[0],batch[1],batch[2]
                    inputs, targets = inputs.to(self._device), targets.to(self._device).to(torch.float32)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    logits = self._network(inputs)["logits"]
                    #targets = targets
                    #print(logits.dtype)
                    #print(targets.dtype)

                    #logits_pred = logits.sigmoid()
                    loss = F.binary_cross_entropy_with_logits(logits, targets)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    elapsed += time.time() - t0
                    losses += loss.item()

                    with torch.no_grad():
                        preds = torch.sigmoid(logits.detach()) # > 0.5
                    all_preds.append(preds.cpu())
                    all_targets.append(targets.cpu())

                    #self.val_preds.append(preds.cpu())
                    #self.val_targets.append(y.cpu())
                    #ap_macro = average_precision_score(targets.cpu(),preds.cpu(),average='macro')
                    #ap_micro = average_precision_score(targets.cpu(),preds.cpu(),average='micro')
                    
                    #_, preds = torch.max(logits, dim=1)
                    #correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                    #total += len(targets)
                if scheduler != None:
                    scheduler.step()
                #train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
                self.total_training_time += elapsed
                preds = torch.cat([o for o in all_preds], dim=0)
                targets = torch.cat([o for o in all_targets], dim=0)
                ap_macro = average_precision_score(targets,preds,average='macro')
                ap_micro = average_precision_score(targets,preds,average='micro')

                if epoch % 5 == 0:
                    test_ap_micro, test_ap_macro = self._compute_mAP(self._network, test_loader)
                    info = "Task {}, Epoch {}/{} => Loss {:.3f},  Train_mAP macro/micro {:.2f}/{:.2f}, Test_mAP macro/micro {:.2f}/{:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.args["init_epoch"],
                        losses / len(train_loader),
                        ap_macro,
                        ap_micro,
                        test_ap_macro,
                        test_ap_micro,
                    )
                else:
                    info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_mAP macro {:.2f}, micro {:.2f}".format(
                        self._cur_task,
                        epoch + 1,
                        self.args["init_epoch"],
                        losses / len(train_loader),
                        ap_macro,
                        ap_micro,
                    )

                prog_bar.set_description(info)
                logging.info(info)

    def compute_task_prototype_and_covariance(self, dataloader, device=None, shrinkage=0.05, limit_samples=None):
        """
        Compute mean (prototype) and covariance for one task.
        Stores both prototype and precision matrix (Σ⁻¹).
        """
        #print('compute prototypes...')
        
        device = device or next(self._network.parameters()).device

        self._network.eval()
        self._network.backbone.activate_eval()

        task_specific_features = []
        pretrained_features = []
        D = None
        stats = None
        pt_stats = None


        with torch.no_grad():
            for k, batch in enumerate(dataloader):  # ignore labels, only task-level
                if limit_samples==None or k<limit_samples:
                    #print("k=",k)
                    if self.args['dataset']!='bigearthnet':
                        (_, x, _) = batch[0],batch[1],batch[2]
                    else:
                        (x, _, _) = batch[0],batch[1],batch[2]
                    #print(x.shape)
                    x = x.to(device)
                    if self.use_task_prototypes:

                        batch_size = x.shape[0]
                        if self.args.get('efficient_inference',False):
                            helper_weights_pt = torch.zeros(batch_size, self._cur_task+1).to(self._device)
                        else:
                            helper_weights_pt = torch.zeros(self._cur_task+1).to(device)


                        self._network.backbone.set_internal_tt_weights(helper_weights_pt)#reset_internal_tt_weights()#set_internal_tt_weights()

                        feats = self._network.forward(x)["features"]
                        #feats = feats.detach()

                        if self.random_projection:
                            print('use random projection for pt')
                            feats = self._random_project(feats, nonlinear='relu')

                        if self.tukey:
                                print('use tukey power transform')
                                feats = torch.pow(feats,exponent=0.5)

                        # use task-specific lora for ts-map feature comp, otherwise use pt model for feature comp
                        if self.args.get('task_specific_prototypes',False):
                            helper_weights_ts = torch.zeros(self._cur_task+1).to(device)
                            helper_weights_ts[self._cur_task] = 1
                            self._network.backbone.set_internal_tt_weights(helper_weights_ts)
                            pretrained_feats = feats.detach()
                            feats = self._network.forward(x)["features"]
                            #feats = feats.detach()
                            if self.random_projection:
                                print('use random projection for pt')
                                feats = self._random_project(feats, nonlinear='relu')
                            if self.tukey:
                                print('use tukey power transform')
                                feats = torch.pow(feats,exponent=0.5)
                            #task_specific_features.append(feats.detach())
                            
                        #print(feats)
                    else: #### not neaded actually
                        feats = self._network(x)["features"]
                        #feats = feats.detach()
                    #all_features.append(feats)

                    if stats is None:
                        D = feats.size(1)
                        #print(D)
                        if self.args.get('diag_cov',False):
                            stats = RunningDiagStats(D, device)
                        else:
                            stats = RunningFullStats(D, device)

                    stats.update(feats)

                    if self.args.get('task_specific_prototypes',False):
                        if pt_stats is None:
                            D = feats.size(1)
                            #print(D)
                            if self.args.get('diag_cov',False):
                                pt_stats = RunningDiagStats(D, device)
                            else:
                                pt_stats = RunningFullStats(D, device)

                        pt_stats.update(pretrained_feats)
                    

        #all_features = torch.cat(all_features, dim=0)  # [N, D]
        if self.use_task_prototypes:
            if self.args.get('task_specific_prototypes',False):

                mean_vec, cov_or_var = stats.finalize()
                pt_mean_vec, _ = pt_stats.finalize()

                self.task_prototypes_residuals[self._cur_task] = mean_vec.squeeze().to(device) - pt_mean_vec.squeeze().to(device) 

                if self.args.get('effective_dim', False):
                    effective_dim = effective_dimension(cov=cov_or_var, lambda_=shrinkage)

                if self.args.get('diag_cov',False):
                    var = shrink_diag(cov_or_var, shrinkage)
                    precision = 1.0 / (var + 1e-8)
                else:
                    cov = shrink_full(cov_or_var, shrinkage) + 1e-8 * torch.eye(D, device=cov.device)  # jitter
                    precision = torch.linalg.pinv(cov)

                    if self.args.get('shared_cov',False):
                        var, cor =  cov_to_var_and_corr(cov)
                        if self._cur_task==0:
                            self.shared_cov = precision
                        else:
                            self.shared_cov = self.shared_cov * ((self._cur_task-1)/self._cur_task) + precision * 1/self._cur_task

                # Save
                self.task_prototypes[self._cur_task] = mean_vec.squeeze().to(device)
                self.task_precisions[self._cur_task] = precision.to(device)
                if self.args.get('effective_dim', False):
                    self.task_effective_dim[self._cur_task] = effective_dim
            else:

                mean, cov_or_var = stats.finalize()
                if self.args.get('shared_cor',False):
                    #R, sigma = cov_to_corr(cov_or_var)
                    if self._cur_task == 0:
                        d = cov_or_var.shape[0]
                        R_sum = torch.zeros((d, d), device=cov_or_var.device, dtype=cov_or_var.dtype)
                    else:
                        R_sum = self.R_sum
                    #cov_or_var = shrink_full(cov_or_var, shrinkage) + 1e-8 * torch.eye(D, device=cov_or_var.device)
                    _, shared_L, sigma_t, R_sum_new, R_t = update_shared_corr_equal_tasks(cov_or_var,R_sum_old= R_sum, num_tasks=self._cur_task+1)
                    self.R_sum = R_sum_new
                    self.task_R[self._cur_task] = R_t
                    self.task_prototypes[self._cur_task] = mean.squeeze().to(device)
                    self.task_precisions[self._cur_task] = sigma_t
                    self.shared_L = shared_L

                    shared_R = (R_sum_new/(self._cur_task+1))
                    for i in range(self._cur_task+1):
                        norm = torch.norm(shared_R-self.task_R[i], p='fro')
                        print("cor norm shared to ",i," ", norm)

                elif self.args.get('shared_pre',False):
                    if self._cur_task == 0:
                        d = cov_or_var.shape[0]
                        P_sum = torch.zeros((d, d), device=cov_or_var.device, dtype=cov_or_var.dtype)
                        #num_tasks = 0
                    else:
                        P_sum = self.P_sum
                    # for each task:
                    #num_tasks += 1
                    #cov_or_var = shrink_full(cov_or_var, shrinkage) + 1e-8 * torch.eye(D, device=cov_or_var.device) 
                    P_shared, Lp, d_t, P_sum, P_t, P_shared_reg = update_shared_precision_corr_equal_tasks(
                        cov_or_var,
                        P_sum_old=P_sum,
                        num_tasks=self._cur_task+1,
                        cov_ridge=1e-6,
                        shrink=1e-3
                    )
                    self.P_sum = P_sum
                    self.task_precisions[self._cur_task]=d_t
                    self.task_P[self._cur_task]=P_t
                    self.task_prototypes[self._cur_task] = mean.squeeze().to(device)
                    self.shared_P = P_shared #_reg
        

                elif self.args.get('shared_eigenvectors', False):

                    self.cov_sum = update_shared_cov_sum(cov_or_var.to(self._device), self.cov_sum)
                    self.shared_U = compute_shared_U_from_cov_sum(self.cov_sum, self._cur_task+1, 64)
                    self.task_eigenvalues[self._cur_task] = task_lam_in_shared_basis(cov_or_var,self.shared_U)
                    self.task_prototypes[self._cur_task] = mean.squeeze().to(device)
                    self.task_precisions[self._cur_task]=self.task_eigenvalues[self._cur_task]

                else:
                    if self.args.get('diag_cov',False):
                        var = shrink_diag(cov_or_var, shrinkage)
                        precision = 1.0 / (var + 1e-8)
                    else:
                        D = cov_or_var.shape[0]
                        cov = shrink_full(cov_or_var, shrinkage) + 1e-8 * torch.eye(D, device=cov_or_var.device)  # jitter
                        precision = torch.linalg.pinv(cov)

                    self.task_prototypes[self._cur_task] = mean.squeeze().to(device)
                    self.task_precisions[self._cur_task] = precision.to(device)


    def compute_task_prototype_and_covariance_class_based(self, dataloader, device=None, shrinkage=0.05):
        print('compute class-based prototypes...')

        device = device or next(self._network.parameters()).device

        self._network.eval()
        self._network.backbone.activate_eval()

        D = None
        class_means = {}   # c -> RunningMean
        class_stats = {}   # c -> RunningDiagStats or RunningFullStats
        class_counts = {}  # c -> int

        with torch.no_grad():
            for _, batch in enumerate(dataloader):
                if self.args['dataset'] != 'bigearthnet':
                    (_, x, y) = batch[0], batch[1], batch[2]
                else:
                    (x, y, _) = batch[0], batch[1], batch[2]

                x = x.to(device)
                y = y.to(device)

                if y.dim() == 1:
                    sample_labels = [torch.tensor([int(label.item())], device=device) for label in y]
                elif y.dim() == 2:
                    sample_labels = [torch.nonzero(label_vec > 0, as_tuple=False).flatten() for label_vec in y]
                else:
                    raise ValueError('class_based_prototypes expects targets of shape [B] or [B, C].')

                if self.use_task_prototypes:
                    if self.args.get('efficient_inference', False):
                        helper_weights_pt = torch.zeros(x.shape[0], self._cur_task + 1).to(device)
                    else:
                        helper_weights_pt = torch.zeros(self._cur_task + 1).to(device)
                    self._network.backbone.set_internal_tt_weights(helper_weights_pt)
                    feats = self._network.forward(x)["features"]

                    if self.random_projection:
                        print('use random projection for pt')
                        feats = self._random_project(feats, nonlinear='relu')

                    if self.tukey:
                        print('use tukey power transform')
                        feats = torch.pow(feats, exponent=0.5)
                else:
                    feats = self._network(x)["features"]

                if D is None:
                    D = feats.size(1)

                for feat, labels in zip(feats, sample_labels):
                    if labels.numel() == 0:
                        continue

                    for c in labels:
                        c_int = int(c.item())
                        if c_int not in class_means:
                            class_means[c_int] = RunningMean(D, device)
                            if self.args.get("diag_cov", False):
                                class_stats[c_int] = RunningDiagStats(D, device)
                            else:
                                class_stats[c_int] = RunningFullStats(D, device)
                            class_counts[c_int] = 0

                        class_means[c_int].update(feat.unsqueeze(0))
                        class_stats[c_int].update(feat.unsqueeze(0))
                        class_counts[c_int] += 1

        labels = sorted(class_means.keys())
        if len(labels) == 0:
            raise RuntimeError('No class samples were found to compute class-based prototypes/covariances.')

        prototypes = []
        precisions = []
        class_stats_dict = {}

        for c in labels:
            mu_c = class_means[c].mean
            n_c = class_stats[c].n

            if self.args.get("diag_cov", False):
                if n_c < 2:
                    precision_c = torch.ones(D, device=device)
                else:
                    _, var_c = class_stats[c].finalize()
                    var_c = shrink_diag(var_c, shrinkage)
                    precision_c = 1.0 / (var_c + 1e-8)
                cov_store = None
            else:
                if n_c < 2:
                    cov_c = torch.eye(D, device=device)
                else:
                    _, cov_c = class_stats[c].finalize()
                    cov_c = shrink_full(cov_c, shrinkage) + 1e-8 * torch.eye(D, device=device)
                precision_c = torch.linalg.pinv(cov_c)
                cov_store = cov_c

            prototypes.append(mu_c)
            precisions.append(precision_c)
            class_stats_dict[c] = {
                "prototype": mu_c.detach().clone(),
                "covariance": None if cov_store is None else cov_store.detach().clone(),
                "inv_cov": precision_c.detach().clone(),
                "count": class_counts[c],
            }

        prototypes = torch.stack(prototypes, dim=0).to(device)
        precisions = torch.stack(precisions, dim=0).to(device)
        labels_tensor = torch.tensor(labels, device=device, dtype=torch.long)

        print(labels_tensor)
        print(prototypes.shape)

        self.task_prototypes[self._cur_task] = prototypes
        self.task_precisions[self._cur_task] = precisions
        self.class_prototypes[self._cur_task] = class_stats_dict
        self.all_class_means[self._cur_task] = {c: class_stats_dict[c]["prototype"] for c in labels}


    
    def compute_class_prototypes_and_covariances(self, dataloader, device=None, eps=1e-8):
        """
        Computes class-level prototypes and covariance matrices for a given dataloader.

        Args:
            backbone (nn.Module): Feature extractor network (e.g., frozen backbone).
            dataloader (DataLoader): Dataloader returning (x, y, task_id) or (x, y).
            device (torch.device, optional): Device to use for computation.
            eps (float): Small regularization term added to covariance diagonals.

        Returns:
            dict: {
                class_id: {
                    "prototype": Tensor(D),
                    "covariance": Tensor(D, D)
                },
                ...
            }
        """
        device = device or next(self._network.parameters()).device

        backbone = self._network
        backbone.eval()

        # Collect all features per class
        features_by_class = defaultdict(list)

        for _, (_, x, y) in enumerate(dataloader):
            #if len(batch) == 3:
            #    x, y, _ = batch
            #else:
            #    x, y = batch
            x, y = x.to(device), y.to(device)

            feats = backbone.forward(x)["features"]  # shape: [B, D]
            feats = feats.detach()
            if self.tukey:
                print('use tukey power transform')
                feats = torch.pow(feats,exponent=0.5)

            for f, label in zip(feats, y):
                features_by_class[int(label.item())].append(f.cpu())

        # Compute prototype and covariance for each class
        class_stats = {}
        for cls, feat_list in features_by_class.items():
            feats = torch.stack(feat_list)  # [N_cls, D]
            proto = feats.mean(dim=0)
            diffs = feats - proto

            # Covariance (regularized)
            cov = (diffs.T @ diffs) / (len(feats) - 1)
            if self.normalization:
                cov = normalize_covariance_to_correlation(cov)

            cov = cov + eps * torch.eye(cov.shape[0], device=cov.device)
            precision = torch.linalg.inv(cov)

            class_stats[cls] = {
                "prototype": proto,
                "covariance": cov,
                "inv_cov": precision
            }

        self.class_prototypes[self._cur_task] = class_stats


    def compute_task_prototype_within_class_cov(self, dataloader, device=None):
        """
        Compute a task-level prototype and pooled within-class covariance matrix.
        This version avoids mixing between-class variation.
        """

        device = device or next(self._network.parameters()).device

        self._network.eval()

        all_features, all_labels = [], []

        with torch.no_grad():
            for _, (_, x, y) in enumerate(dataloader):
                x = x.to(device)
                #feats = self.backbone(x)
                #all_features.append(feats)
                all_labels.append(y.to(device))

                

                helper_weights_pt = torch.zeros(self._cur_task+1).to(device)

                        
                self._network.backbone.set_internal_tt_weights(helper_weights_pt)#reset_internal_tt_weights()#set_internal_tt_weights()

                feats = self._network.forward(x)["features"]
                all_features.append(feats)
                        

        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # --- Step 1: compute class-wise means ---
        class_means = {}
        for cls in all_labels.unique():
            cls_mask = (all_labels == cls)
            class_feats = all_features[cls_mask]
            class_means[int(cls.item())] = class_feats.mean(dim=0)

        # --- Step 2: compute pooled within-class covariance ---
        centered_features = []
        for cls in all_labels.unique():
            cls_mask = (all_labels == cls)
            feats = all_features[cls_mask]
            mu_c = class_means[int(cls.item())]
            centered_features.append(feats - mu_c)

        centered_features = torch.cat(centered_features, dim=0)
        # Covariance = (X^T X) / (N - C)
        cov = torch.cov(centered_features.T)  # (D, D)
        # Add small regularization for numerical stability
        cov += 1e-5 * torch.eye(cov.size(0), device=device)

        # --- Step 3: compute the task prototype (mean of class means) ---
        task_proto = torch.stack(list(class_means.values())).mean(dim=0)

        # --- Step 4: store results ---
        # task_stats = {
        #     "prototype": task_proto.detach(),
        #     "covariance": cov.detach(),
        #     "class_means": class_means  # optional: useful for debugging/visualization
        # }
        self.task_prototypes[self._cur_task] = task_proto.detach().to(device)
        self.task_precisions[self._cur_task] = cov.detach().to(device)
        self.all_class_means[self._cur_task] = class_means

 


   

    def aggregate_class_distances_to_task_distances(self, class_distance_list, strategy='min'):
        """Aggregate per-class distances into one distance per task.

        Args:
            class_distance_list: list of tensors, each [B, C_t] for one task.
            strategy: min|mean|median|topk_mean|softmin

        Returns:
            Tensor [B, T] with one distance per task.
        """
        if not isinstance(class_distance_list, list) or len(class_distance_list) == 0:
            raise ValueError('class_distance_list must be a non-empty list of tensors')

        aggregated = []
        for d in class_distance_list:
            if d.dim() == 1:
                d = d.unsqueeze(0)

            if strategy == 'min':
                task_d = d.min(dim=1).values
            elif strategy in ['mean', 'avg']:
                task_d = d.mean(dim=1)
            elif strategy == 'median':
                task_d = d.median(dim=1).values
            elif strategy == 'topk_mean':
                k = max(1, int(self.args.get('class_agg_topk', 3)))
                k = min(k, d.size(1))
                task_d = d.topk(k, dim=1, largest=False).values.mean(dim=1)
            elif strategy == 'softmin':
                tau = float(self.args.get('class_agg_temperature', 0.5))
                tau = max(tau, 1e-6)
                task_d = -tau * torch.logsumexp(-d / tau, dim=1)
            else:
                raise ValueError(f'Unknown class_to_task_agg strategy: {strategy}')

            aggregated.append(task_d)

        return torch.stack(aggregated, dim=1)

    def aggregate_class_distances_to_task_weights(self, class_distance_list, class_agg_strategy='min', weighting_strat='softmax'):
        """Aggregate per-class distances directly into per-task weights."""
        if class_agg_strategy in ['same_class_vote', 'class_vote', 'per_class_vote']:
            if not isinstance(class_distance_list, list) or len(class_distance_list) == 0:
                raise ValueError('class_distance_list must be a non-empty list of tensors')

            prepared = []
            min_classes = None
            for d in class_distance_list:
                if d.dim() == 1:
                    d = d.unsqueeze(0)
                prepared.append(d)
                if min_classes is None:
                    min_classes = d.size(1)
                else:
                    min_classes = min(min_classes, d.size(1))

            # Build [B, T, C] distances where C is shared classes across tasks.
            stacked = torch.stack([d[:, :min_classes] for d in prepared], dim=1)
            B, T, C = stacked.shape

            # Each class votes for the task with minimum distance among same class prototypes.
            winners = torch.argmin(stacked, dim=1)  # [B, C]
            votes = F.one_hot(winners, num_classes=T).sum(dim=1).to(stacked.dtype)  # [B, T]
            task_weights = votes / (votes.sum(dim=1, keepdim=True) + 1e-12)

            # Keep a distance-like tensor for stats/debug downstream.
            distances_for_weights = 1.0 - task_weights

            if not self.args.get('efficient_inference', False) and task_weights.dim() == 2 and task_weights.size(0) == 1:
                task_weights = task_weights.squeeze(0)
                distances_for_weights = distances_for_weights.squeeze(0)

            return task_weights, distances_for_weights

        task_distances = self.aggregate_class_distances_to_task_distances(
            class_distance_list=class_distance_list,
            strategy=class_agg_strategy,
        )

        distances_for_weights = task_distances
        if not self.args.get('efficient_inference', False) and distances_for_weights.dim() == 2 and distances_for_weights.size(0) == 1:
            distances_for_weights = distances_for_weights.squeeze(0)

        if self.args.get('distance', 'mahalanobis') == 'gaussian_likelihood' and self.args.get('diki_analog', False):
            task_weights = self.get_task_weights(distances=distances_for_weights, weighting_strat='max')
        elif self.args.get('distance', 'mahalanobis') == 'gaussian_likelihood':
            if distances_for_weights.dim() == 1:
                task_weights = distances_for_weights / (distances_for_weights.sum() + 1e-12)
            else:
                task_weights = distances_for_weights / (distances_for_weights.sum(dim=1, keepdim=True) + 1e-12)
        else:
            task_weights = self.get_task_weights(
                distances=distances_for_weights,
                weighting_strat=weighting_strat,
            )

        return task_weights, distances_for_weights

    def compute_distance_to_prototypes(self, features, prototypes, precisions, distance_metric='cosine'):

        distances = None
        
        # if not self.args.get('class_based_prototypes', False):
        #     for i in range(self._cur_task+1):
                    
        #             if distance_metric == 'mahalanobis':
        #                 if self.args.get('shared_cor', False):
        #                     dist = mahalanobis_sq_batch(X=features, Y=self.task_prototypes[i],L=self.shared_L,sigma_t=self.task_var[i])
        #                     distances.append(torch.sqrt(dist))
        #                 elif self.args.get('shared_pre', False):
        #                     dist = mahalanobis_sq_paired_batch_from_precision_corr(X=features, Y=self.task_prototypes[i], P_shared=self.shared_P, d_t=self.task_prec_diag[i])
        #                     distances.append(torch.sqrt(dist))
        #                 else: distances.append(mahalanobis_distance(features, self.task_prototypes[i], self.task_covariances[i],diagonal=self.args.get('diag_cov',False)))
                    
        #             if distance_metric == 'cosine':
        #                 distances.append(1 - torch.nn.functional.cosine_similarity(features,self.task_prototypes[i]))
        #             if distance_metric == 'L2':
        #                 #print('l2')
        #                 distances.append(torch.linalg.norm(features - self.task_prototypes[i], ord=2, dim=1))
        #             if distance_metric == 'L1':
        #                 distances.append(torch.linalg.norm(features - self.task_prototypes[i], ord=1, dim=1))


        if not self.args.get('class_based_prototypes', False):
                    
            if distance_metric == 'mahalanobis':
                if self.args.get('shared_cor', False):
                    dist = mahalanobis_sq_batch_all_tasks_taskmeans(X=features, Y_all=prototypes, sigma_all=precisions, L=self.shared_L)
                    distances = torch.sqrt(dist)
                elif self.args.get('shared_pre', False):
                    distances = mahalanobis_sq_to_tasks_from_precision_corr(X=features, Y=prototypes, P_reg=self.shared_P, d_all=precisions)
                    #distances.append(torch.sqrt(dist))
                    distances = torch.sqrt(distances)
                elif self.args.get('shared_eigenvectors',False):
                    lam_all = torch.stack([self.task_eigenvalues[i] for i in range(self._cur_task+1)], dim=0)
                    distances = mahalanobis_sq_to_tasks_sharedU(X=features, Y_all=prototypes, U=self.shared_U, lam_all=lam_all)
                    #distances.append(torch.sqrt(dist))
                    distances = torch.sqrt(distances)
                else: distances = mahalanobis_distance_tasks(features, prototypes, precisions,sqrt=True,diagonal_precision=self.args.get('diag_cov',False))
            
            if distance_metric == 'gaussian_likelihood':
                distances = gaussian_log_likelihood_tasks(features, prototypes, precisions, diagonal_precision=self.args.get('diag_cov',False))

            if distance_metric == 'cosine':
                #distances = (1 - torch.nn.functional.cosine_similarity(features,prototypes))
                similarities = torch.nn.functional.cosine_similarity(
                    features[:, None, :],      # [N, 1, D]
                    prototypes[None, :, :],    # [1, M, D]
                    dim=-1
                )
                distances = 1 - similarities
            if distance_metric == 'L2':
                #print('l2')
                distance_vec = features[:, None, :] - prototypes[None, :, :]
                distances = torch.linalg.norm(distance_vec, ord=2, dim=-1)
            if distance_metric == 'L1':
                distance_vec = features[:, None, :] - prototypes[None, :, :]
                distances = torch.linalg.norm(distance_vec, ord=1, dim=-1)

        else:
            distances = []
            for i in range(self._cur_task+1):
                proto_i = self.task_prototypes[i]
                prec_i = self.task_precisions[i]

                if distance_metric == 'mahalanobis':
                    if prec_i.dim() == 3:
                        class_dists = mahalanobis_distance_classes(features, proto_i, prec_i, sqrt=True)
                    elif prec_i.dim() == 2 and self.args.get('diag_cov', False):
                        diff = features[:, None, :] - proto_i[None, :, :]
                        class_dists = torch.sqrt((diff * diff * prec_i.unsqueeze(0)).sum(dim=-1) + 1e-8)
                    else:
                        class_dists = mahalanobis_distance(features, proto_i, prec_i, diagonal=False)
                    distances.append(class_dists)

                if distance_metric == 'cosine':
                    sim = torch.nn.functional.cosine_similarity(features[:, None, :], proto_i[None, :, :], dim=-1)
                    distances.append(1 - sim)

                if distance_metric == 'L2':
                    d = torch.linalg.norm(features[:, None, :] - proto_i[None, :, :], ord=2, dim=-1)
                    distances.append(d)

                if distance_metric == 'L1':
                    d = torch.linalg.norm(features[:, None, :] - proto_i[None, :, :], ord=1, dim=-1)
                    distances.append(d)

            return distances
        #print(distances)
        #if len(distances)==1:
        #    return torch.cat(distances)
        #else:
        #print('new_distances: ', distances)
        return distances.squeeze(0) #return torch.cat(distances)#.squeeze()
    
    def compute_distance_to_task_specific_prototypes(self, all_features, distance_metric='mahalanobis'):

        distances = []
        

        for i in range(self._cur_task+1):
                
                if distance_metric == 'mahalanobis':
                    distances.append(mahalanobis_distance(all_features[i], self.task_prototypes[i], self.task_precisions[i]))
                if distance_metric == 'cosine':
                    distances.append(1 - torch.nn.functional.cosine_similarity(all_features[i],self.task_prototypes[i]))
                if distance_metric == 'L2':
                    distances.append(torch.linalg.norm(all_features[i] - self.task_prototypes[i], ord=2))
                if distance_metric == 'L1':
                    distances.append(torch.linalg.norm(all_features[i] - self.task_prototypes[i], ord=1))

        #print(distances)
        #if len(distances)==1:
        #    return torch.cat(distances)
        #else:
        return torch.cat(distances)#.squeeze()
    
    def get_task_weights(self, distances, weighting_strat='softmax'):
        if weighting_strat == 'min_plus_random':
            min_distance_task_id = torch.argmin(distances)
                #def get_task_weights(self, sample_task_id, current_task):
            random_task = random.randint(0, self._cur_task)
            random_task2 = random.randint(0, self._cur_task)
            weights = [0 for i in range(self._cur_task+1)]
            weights[random_task] = 1
            weights[random_task2] = 1
            weights[min_distance_task_id] = 1
            weights = torch.Tensor(weights)
            weights = weights/(weights.sum()+1e-8)


        elif weighting_strat == 'min':
            min_distance_task_id = torch.argmin(distances)
                #def get_task_weights(self, sample_task_id, current_task):
            #random_task = random.randint(0, self.current_task)
            weights = [0 for i in range(self._cur_task+1)]
            #weights[random_task] = 1
            weights[min_distance_task_id] = 1
            weights = torch.Tensor(weights)
            weights = weights/(weights.sum()+1e-9)

        elif weighting_strat == 'max':
            indices = torch.argmax(distances, dim=1)
            weights = F.one_hot(indices, num_classes=distances.shape[1]).to(distances.dtype)
        
        elif weighting_strat == 'weighted_merge':
            #min_distance_task_id = torch.argmin(distances)
                #def get_task_weights(self, sample_task_id, current_task):
            #random_task = random.randint(0, self.current_task)
            weights = [1 for i in range(self._cur_task+1)]
            #weights[random_task] = 1
            #weights[min_distance_task_id] = 1
            weights = torch.Tensor(weights)
            weights = weights/(weights.sum()+1e-9)
            
        elif weighting_strat == 'softmax':
            temperature = self.args.get('temperature', 0.5)
            if self.args.get('efficient_inference'):
                weights = torch.softmax(-distances / temperature, dim=1)
            else:
                weights = torch.softmax(-distances / temperature, dim=0)

        elif weighting_strat == 'dynamic_softmax':
            
            temperature, margin = self.compute_relative_margin_taus(dists=distances)
            print(temperature)
            print(margin)
            weights = torch.softmax(-distances / temperature, dim=0)

        elif weighting_strat == 'margin_softmax':
            
            temperature = self.compute_margin_taus(dists=distances)

            weights = torch.softmax(-distances / temperature, dim=0)

        elif weighting_strat == 'test_dyn':
            
            sorted_vals, _ = torch.sort(distances)
            temperature = 0.5
            if len(sorted_vals)>1:  
                d1, d2 = sorted_vals[0], sorted_vals[1]
                margin = (d2 - d1)
                if margin > 2.5:
                    temperature = 0.5
                else: temperature = 10
            #temperature, margin = self.compute_relative_margin_taus(dists=distances)
            print(temperature)
            #print(margin)
            weights = torch.softmax(-distances / temperature, dim=0)

        return weights
    
    def compute_relative_margin_taus(self, dists, mode="exp", 
                                 tau_min=0.05, tau_max=10.0, 
                                 beta=3.0, gamma=10.0, m0=0.5, alpha=5.0, eps=1e-8):
        """
        dists: [B, T] matrix of Mahalanobis distances
        mode: 'exp', 'sigmoid', or 'inv'
        Returns:
            taus: [B] tensor of temperatures
            margins_rel: [B] relative margins
        """

        
        sorted_vals, _ = torch.sort(dists)
        if len(sorted_vals)>1:
            d1, d2 = sorted_vals[0], sorted_vals[1]
            margin_rel = (d2 - d1) / (d1 + eps)
            margin_rel = margin_rel.clamp(min=0.0)

            if mode == "exp":
                taus = tau_min + (tau_max - tau_min) * torch.exp(-beta * margin_rel)
            elif mode == "sigmoid":
                taus = tau_min + (tau_max - tau_min) / (1 + torch.exp(gamma * (margin_rel - m0)))
            elif mode == "inv":
                taus = tau_max / (1 + alpha * margin_rel)
                taus = torch.clamp(taus, min=tau_min, max=tau_max)
            else:
                raise ValueError(f"Unknown mode {mode}")
        else:
            taus = tau_min
            margin_rel=0

        return taus, margin_rel
    
    def compute_margin_taus(self, dists, 
                                 tau_min=0.01, gamma=1.0, eps=1e-8):
        """
        dists: [B, T] matrix of Mahalanobis distances
        Returns:
            taus: [B] tensor of temperatures
            margins_rel: [B] relative margins
        """

        
        sorted_vals, _ = torch.sort(dists)
        if len(sorted_vals)>1:
            d1, d2 = sorted_vals[0], sorted_vals[1]
            margin = (d2 - d1)
            margin = margin.clamp(min=0.0)
            taus = margin * gamma

            
        else:
            taus = tau_min


        return taus

    def _random_project(self, feats, nonlinear=None):
        """Apply fixed random projection + nonlinearity."""
        #print('features device', feats.device)
        #print('projection device', self.random_matrix.device)
        projected = feats @ self.random_matrix.T
        if nonlinear=='relu':
            projected = F.relu(projected)  # you can swap with GELU or tanh
        if nonlinear == 'squared':
            projected = projected**2
        return projected
    
    def compute_weight_stats(self):
        """
        heads: dict mapping task_id -> (W, b)
            W: weight matrix of shape [num_classes, feat_dim]
            b: bias vector of shape [num_classes] or None
        """
        head = self._network.fc
        num_tasks = self._cur_task
        heads = {}
        inc = int(len(head.weight.data) / num_tasks)
        for i in range(num_tasks):
            weight = head.weight.data[i*inc:(i+1)*inc] 
            bias = head.bias.data[i*inc:(i+1)*inc]
            heads[i] = (weight,bias)
        stats = {}

        for task_id, (W, b) in heads.items():
            # move to CPU for simplicity
            W = W.detach().cpu()

            # compute per-class weight norms
            norms = torch.norm(W, p=2, dim=1).numpy()  # shape [num_classes]

            # compute summary statistics
            stats[task_id] = {
                "per_class_norms": norms.tolist(),              # array, per class
                "mean_norm": float(norms.mean()),      # scalar
                "std_norm": float(norms.std()),        # scalar
                "min_norm": float(norms.min()),        # scalar
                "max_norm": float(norms.max()),        # scalar
            }

            # bias statistics (optional)
            if b is not None:
                b = b.detach().cpu().numpy()
                stats[task_id]["bias_mean"] = float(b.mean())
                stats[task_id]["bias_std"]  = float(b.std())
            else:
                stats[task_id]["bias_mean"] = None
                stats[task_id]["bias_std"]  = None

        return stats


def update_shared_cov_sum(Sigma_t: torch.Tensor, S_sum: torch.Tensor):
    # keep it symmetric
    Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)
    return S_sum + Sigma_t

def compute_shared_U_from_cov_sum(S_sum: torch.Tensor, num_tasks: int, k: int, shrink: float = 1e-3):
    """
    Computes shared eigenvectors U from average covariance.
    """
    S_avg = S_sum / float(num_tasks)
    S_avg = 0.5 * (S_avg + S_avg.T)
    d = S_avg.shape[0]
    S_avg = (1.0 - shrink) * S_avg + shrink * torch.eye(d, device=S_avg.device, dtype=S_avg.dtype)

    evals, evecs = torch.linalg.eigh(S_avg)  # ascending
    U = evecs[:, -k:]                        # (d,k) top-k
    return U

def task_lam_in_shared_basis(Sigma_t: torch.Tensor, U: torch.Tensor, eps: float = 1e-8):
    """
    lam_t = diag(U^T Sigma_t U)
    """
    Sigma_t = 0.5 * (Sigma_t + Sigma_t.T)
    G = U.T @ Sigma_t @ U
    lam_t = torch.diagonal(G).clamp_min(eps)
    return lam_t  # (k,)

def shrink_full(cov, alpha, mode="diag"):
    D = cov.shape[0]

    if mode == "identity":
        mu = torch.trace(cov) / D
        target = mu * torch.eye(D, device=cov.device)
    elif mode == "diag":
        target = torch.diag(torch.diag(cov))
    else:
        raise ValueError

    cov = (1 - alpha) * cov + alpha * target
    #cov = cov 
    return cov

def shrink_diag(var, alpha):
    target = var.mean()
    return (1 - alpha) * var + alpha * target

def shrink_correlation(R: torch.Tensor, lam: float):
    """
    lam in [0,1], e.g. 0.05 - 0.2
    """
    d = R.size(0)
    I = torch.eye(d, device=R.device, dtype=R.dtype)
    R_shrunk = (1 - lam) * R + lam * I
    return R_shrunk

class RunningDiagStats:
    def __init__(self, D, device):
        self.n = 0
        self.mean = torch.zeros(D, device=device)
        self.M2 = torch.zeros(D, device=device)  # per-dimension

    def update(self, x):  # x: (B, D)
        B = x.size(0)
        batch_mean = x.mean(dim=0)
        batch_var = ((x - batch_mean) ** 2).sum(dim=0)

        delta = batch_mean - self.mean
        new_n = self.n + B

        self.mean += delta * B / new_n
        self.M2 += batch_var + delta**2 * self.n * B / new_n
        self.n = new_n

    def finalize(self):
        var = self.M2 / (self.n - 1)
        return self.mean, var


class RunningFullStats:
    def __init__(self, D, device):
        self.n = 0
        self.mean = torch.zeros(D, device=device)
        self.M2 = torch.zeros(D, D, device=device)

    def update(self, x):  # x: (B, D)
        B = x.size(0)
        batch_mean = x.mean(dim=0)
        xc = x - batch_mean

        delta = batch_mean - self.mean
        new_n = self.n + B

        self.mean += delta * B / new_n
        self.M2 += xc.T @ xc + torch.outer(delta, delta) * (self.n * B / new_n)
        self.n = new_n

    def finalize(self):
        cov = self.M2 / (self.n - 1)
        return self.mean, cov
    
class RunningMean:
    def __init__(self, D, device):
        self.n = 0
        self.mean = torch.zeros(D, device=device)

    def update(self, x):  # x: [B, D]
        B = x.size(0)
        new_n = self.n + B
        delta = x.mean(dim=0) - self.mean
        self.mean += delta * (B / new_n)
        self.n = new_n


def effective_dimension(cov, lambda_):
    """
    cov: (d, d) torch.Tensor, symmetric
    lambda_: float or scalar tensor
    """
    if cov.dim() == 1:
        eigvals = cov
        print('eff_dim of cov_diag')
    elif cov.dim() == 2:
        # Full covariance: compute eigenvalues
        eigvals = torch.linalg.eigvalsh(cov)
        print('eff_dim of cov_full')

    # Clamp small negatives due to numerical noise
    eigvals = torch.clamp(eigvals, min=0.0)

    d_eff = torch.sum(eigvals / (eigvals + lambda_ + 1e-8)) ### 1e-6 for numerical stability in case of no shrinkage

    return d_eff.item()

    


def mahalanobis_distance(x, mu, precision, diagonal=False):
    """
    Computes squared Mahalanobis distances.

    x: Tensor [D] or [B, D]
    mu: Tensor [D] or [C, D]
    precision:
        - diag: [D]
        - full: [D, D]

    Returns:
        distances:
            - [C]     if x is [D]
            - [B, C]  if x is [B, D]
    """

    # --- normalize shapes ---
    if x.dim() == 1:
        x = x.unsqueeze(0)          # [1, D]

    if mu.dim() == 1:
        mu = mu.unsqueeze(0)        # [1, D]

    # x:  [B, D]
    # mu: [C, D]

    diff = x[:, None, :] - mu[None, :, :]   # [B, C, D]

    # --- diagonal precision ---
    #if precision.dim() == 1:
    if diagonal:
        # precision: [D]
        d2 = torch.sum(diff * diff * precision, dim=-1)

    # --- full precision ---
    else:
        # precision: [D, D]
        # (x - μ)^T Σ⁻¹ (x - μ)
        tmp = diff @ precision
        d2 = torch.sum(tmp * diff, dim=-1)

    d2 = torch.sqrt(d2 + 1e-8)

    # squeeze batch dim if needed
    if d2.shape[0] == 1:
        return d2.squeeze(0)

    return d2

def mahalanobis_distance_classes(features, prototypes, inv_covariances, sqrt=True):
    """
    Efficiently compute Mahalanobis distances between feature vectors and class prototypes.

    Args:
        features (Tensor): [B, D] feature vectors (batch of test samples)
        prototypes (Tensor): [C, D] class prototypes
        inv_covariances (Tensor): [C, D, D] inverse covariance matrices for each class

    Returns:
        Tensor: [B, C] Mahalanobis distances (smaller means closer)
    """
    B, D = features.shape
    C = prototypes.shape[0]

    # Expand for broadcasting: (B, C, D)
    diff = features.unsqueeze(1) - prototypes.unsqueeze(0)

    # Compute distances using batch matrix multiplication
    # diff @ inv_cov @ diff^T, efficiently in batch
    temp = torch.einsum("bcd,cde->bce", diff, inv_covariances)  # [B, C, D]
    dists = torch.einsum("bcd,bcd->bc", temp, diff)
    if sqrt:
        dists = dists.clamp_min(1e-9).sqrt()  # [B, C]

    return dists



def mahalanobis_distance_tasks(
    features,
    prototypes,
    precisions,
    sqrt: bool = True,
    diagonal_precision: bool = False,
):
    """
    Efficiently compute Mahalanobis distances between feature vectors and task prototypes.

    Args:
        features (Tensor): [B, D] feature vectors (batch of test samples)
        prototypes (Tensor): [T, D] task prototypes
        precisions (Tensor):
            - if diagonal_precision=False: [T, D, D] inverse covariance per task
            - if diagonal_precision=True:  [T, D] diagonal entries of inverse covariance
        sqrt (bool): if True, return sqrt'd distances
        diagonal_precision (bool): if True, treat precisions as diagonal

    Returns:
        Tensor: [B, T] Mahalanobis distances (smaller means closer)
    """
    B, D = features.shape
    T = prototypes.shape[0]

    # (B, T, D)
    diff = features.unsqueeze(1) - prototypes.unsqueeze(0)

    if diagonal_precision:
        # precisions: (T, D) -> broadcast to (B, T, D)
        # dists = sum_d p_cd * diff_bcd^2
        dists = (diff * diff * precisions.unsqueeze(0)).sum(dim=-1)  # (B, T)
    else:
        # precisions: (T, D, D)
        temp = torch.einsum("bcd,cde->bce", diff, precisions)  # (B, T, D)
        dists = torch.einsum("bcd,bcd->bc", temp, diff)        # (B, T)

    if sqrt:
        dists = dists.clamp_min(1e-9).sqrt()

    return dists

def gaussian_log_likelihood_tasks(
    features,
    prototypes,
    precisions,
    diagonal_precision: bool = False,
):
    """
    Compute multivariate Gaussian log likelihoods for each task.

    Args:
        features (Tensor): [B, D] feature vectors (batch of test samples)
        prototypes (Tensor): [T, D] task means
        precisions (Tensor):
            - if diagonal_precision=False: [T, D, D] inverse covariance per task
            - if diagonal_precision=True:  [T, D] diagonal entries of inverse covariance
        diagonal_precision (bool): if True, treat precisions as diagonal

    Returns:
        Tensor: [B, T] Gaussian log likelihoods
                (larger means more likely)
    """

    B, D = features.shape
    T = prototypes.shape[0]

    # (B,T,D)
    diff = features.unsqueeze(1) - prototypes.unsqueeze(0)

    if diagonal_precision:
        # Mahalanobis quadratic term:
        # (x-mu)^T Sigma^-1 (x-mu)
        mahal_sq = (diff * diff * precisions.unsqueeze(0)).sum(dim=-1)

        # log |Sigma| = - log |Precision|
        log_det_cov = -torch.log(precisions).sum(dim=-1)  # (T,)

    else:
        # quadratic term
        temp = torch.einsum("btd,tde->bte", diff, precisions)
        mahal_sq = torch.einsum("btd,btd->bt", temp, diff)

        # log |Sigma| = - log |Precision|
        sign, logdet_precision = torch.linalg.slogdet(precisions)

        # Should be positive definite, so sign should be +1
        log_det_cov = -logdet_precision  # (T,)

    # Gaussian log probability
    log_prob = (
        -0.5 * mahal_sq
        -0.5 * log_det_cov.unsqueeze(0)
        -0.5 * D * torch.log(torch.tensor(2 * torch.pi, device=features.device))
    )

    return log_prob

def normalize_covariance_to_correlation(Sigma: torch.Tensor) -> torch.Tensor:
    """
    Normalize covariance matrices to correlation form (diagonal = 1).
    Works for batched inputs: Sigma shape = (..., d, d)
    """
    diag = Sigma.diagonal(dim1=-2, dim2=-1)
    diag = torch.clamp(diag, min=1e-12)
    inv_sqrt = torch.diag_embed(1.0 / torch.sqrt(diag))
    Sigma_norm = inv_sqrt @ Sigma @ inv_sqrt
    Sigma_norm = 0.5 * (Sigma_norm + Sigma_norm.transpose(-1, -2))  # enforce symmetry
    return Sigma_norm


def cov_to_var_and_corr(Sigma_t: torch.Tensor, eps: float = 1e-8):
    """
    Sigma_t: (d, d) task covariance matrix

    Returns:
        var_t: (d,) task variances
        R_t:   (d, d) task correlation matrix
    """
    # Variances (diagonal)
    var_t = torch.diag(Sigma_t).clone()

    # Protect against tiny / zero variances
    std_t = torch.sqrt(var_t + eps)

    # Compute correlation: D^{-1} Sigma D^{-1}
    R_t = Sigma_t / std_t[:, None] / std_t[None, :]

    # Force exact unit diagonal (important numerically)
    R_t.fill_diagonal_(1.0)

    return var_t, R_t


def cov_to_corr(Sigma: torch.Tensor, eps: float = 1e-6):
    """
    Sigma: (d, d) covariance (assumed symmetric-ish)
    Returns:
      R: (d, d) correlation
      sigma: (d,) stddevs
    """
    # Ensure symmetry
    Sigma = 0.5 * (Sigma + Sigma.T)

    var = torch.diagonal(Sigma).clamp_min(0.0)  # avoid tiny negative due to numerics
    sigma = torch.sqrt(var + eps)               # (d,)

    inv_sigma = 1.0 / sigma
    # R = D^{-1} Sigma D^{-1} implemented via outer products
    R = Sigma * inv_sigma[:, None] * inv_sigma[None, :]
    # Force exact unit diagonal
    R.fill_diagonal_(1.0)
    # Re-symmetrize (tiny numerical drift)
    R = 0.5 * (R + R.T)
    return R, sigma


def normalize_to_correlation(R: torch.Tensor, eps: float = 1e-12):
    """
    Enforce symmetry + unit diagonal scaling.
    """
    R = 0.5 * (R + R.T)
    d = torch.diagonal(R).clamp_min(eps)
    inv_sqrt_d = 1.0 / torch.sqrt(d)
    R = R * inv_sqrt_d[:, None] * inv_sqrt_d[None, :]
    R.fill_diagonal_(1.0)
    R = 0.5 * (R + R.T)
    return R


def update_shared_corr_ema(
    Sigma_t: torch.Tensor,
    R_old: torch.Tensor,
    alpha: float = 0.05,     # how fast to adapt (smaller = more stable)
    eps: float = 1e-6,
    shrink: float = 1e-3,    # lambda for shrinkage toward I
):
    """
    Given new task covariance and old shared correlation, returns:
      R_new: updated shared correlation (unit diag)
      L: Cholesky factor of regularized R_new (for distances)
      sigma_t: task stds (store per task)
    """
    # 1-2) task correlation + task stds
    R_t, sigma_t = cov_to_corr(Sigma_t, eps=eps)

    # 3) EMA update
    R_mix = (1.0 - alpha) * R_old + alpha * R_t

    # 4) enforce correlation constraints
    R_new = normalize_to_correlation(R_mix)

    # 5) shrinkage to ensure SPD / conditioning
    d = R_new.shape[0]
    R_reg = (1.0 - shrink) * R_new + shrink * torch.eye(d, device=R_new.device, dtype=R_new.dtype)

    # 6) Cholesky (shared factor for distances)
    L = torch.linalg.cholesky(R_reg)

    return R_new, L, sigma_t


def update_shared_corr_equal_tasks(
    Sigma_t: torch.Tensor,
    R_sum_old: torch.Tensor,
    num_tasks: int,
    eps: float = 1e-6,
    shrink: float = 1e-3,
):
    """
    Equal task weighting online update.

    Inputs:
      Sigma_t: (d, d) covariance for current task
      R_sum_old: (d, d) running sum of R_t across tasks so far
      num_tasks_old: number of tasks already included in R_sum_old

    Returns:
      R_new: (d, d) shared correlation (unit diagonal)
      L: (d, d) Cholesky of regularized shared correlation (for distances)
      sigma_t: (d,) task stddevs (store per task)
      R_sum_new: updated running sum
      num_tasks_new: updated task count
    """
    #Sigma_t = shrink_full(Sigma_t, alpha=0.01)
    R_t, sigma_t = cov_to_corr(Sigma_t, eps=eps)

    R_sum_new = R_sum_old + R_t
    #num_tasks_new = num_tasks_old + 1

    R_avg = R_sum_new / float(num_tasks)
    R_new = normalize_to_correlation(R_avg)

    d = R_new.shape[0]
    R_reg = (1.0 - shrink) * R_new + shrink * torch.eye(d, device=R_new.device, dtype=R_new.dtype)

    L = torch.linalg.cholesky(R_reg)

    return R_new, L, sigma_t, R_sum_new, R_t#, num_tasks_new

def mahalanobis_sq_batch(X, Y, sigma_t, L):
    """
    Paired batch distances using shared Cholesky L and task std sigma_t.
    X: (B, d), Y: (B, d) or (d,)
    """
    U = (X - Y) / sigma_t  # (B, d)
    Z = torch.linalg.solve_triangular(L, U.T, upper=False)  # (d, B)
    return (Z * Z).sum(dim=0)


def mahalanobis_sq_batch_all_tasks_taskmeans(X, Y_all, sigma_all, L, chunk_size=None):
    """
    Distances from each x in X to each task mean in Y_all.

    X: (B, d)
    Y_all: (T, d)
    sigma_all: (T, d)
    L: (d, d)

    Returns:
      D2: (B, T)
    """
    B, d = X.shape
    T, d2 = Y_all.shape
    assert d == d2 == sigma_all.shape[1]
    assert sigma_all.shape[0] == T

    if chunk_size is None:
        # diff: (B, T, d)
        diff = X[:, None, :] - Y_all[None, :, :]
        # U: (B, T, d)
        U = diff / sigma_all[None, :, :]
        # RHS: (B, T, d) -> (T, d, B)
        RHS = U.permute(1, 2, 0)  # (T, d, B)
        Z = torch.linalg.solve_triangular(L, RHS, upper=False)  # (T, d, B)
        return (Z * Z).sum(dim=1).transpose(0, 1)  # (B, T)

    # memory-safe chunking over tasks
    outs = []
    for s in range(0, T, chunk_size):
        e = min(T, s + chunk_size)
        Yc = Y_all[s:e]          # (Tc, d)
        Sc = sigma_all[s:e]      # (Tc, d)

        diff = X[:, None, :] - Yc[None, :, :]   # (B, Tc, d)
        U = diff / Sc[None, :, :]               # (B, Tc, d)
        RHS = U.permute(1, 2, 0)                # (Tc, d, B)
        Z = torch.linalg.solve_triangular(L, RHS, upper=False)  # (Tc, d, B)
        outs.append((Z * Z).sum(dim=1).transpose(0, 1))         # (B, Tc)

    return torch.cat(outs, dim=1)

def normalize_to_unit_diagonal(A: torch.Tensor, eps: float = 1e-12):
    """
    Enforce symmetry + unit diagonal scaling:
      A <- D^{-1/2} A D^{-1/2}, diag(A)=1
    Works for both covariance-derived correlations and precision-derived partial correlations.
    """
    A = 0.5 * (A + A.T)
    d = torch.diagonal(A).clamp_min(eps)
    inv_sqrt_d = 1.0 / torch.sqrt(d)
    A = A * inv_sqrt_d[:, None] * inv_sqrt_d[None, :]
    A.fill_diagonal_(1.0)
    A = 0.5 * (A + A.T)
    return A

def cov_to_precision(Sigma: torch.Tensor, ridge: float = 1e-6):
    """
    Lambda = (Sigma + ridge*I)^{-1} via Cholesky.
    More stable than torch.inverse.
    """
    Sigma = 0.5 * (Sigma + Sigma.T)
    d = Sigma.shape[0]
    Sigma_reg = Sigma + ridge * torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
    L = torch.linalg.cholesky(Sigma_reg)
    I = torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)
    Lambda = torch.cholesky_solve(I, L)
    return 0.5 * (Lambda + Lambda.T)

def symmetrize(A: torch.Tensor) -> torch.Tensor:
    return 0.5 * (A + A.T)

def precision_from_cov_pinv(
    Sigma: torch.Tensor,
    ridge: float = 0.0,
    rcond: float | None = None,
    shrink: float | None = None,
    ) -> torch.Tensor:
    """
    Compute precision Lambda ≈ (Sigma + ridge*I)^{+} using Moore-Penrose pseudo-inverse.

    Args:
      Sigma: (d,d) covariance (assumed symmetric-ish).
      ridge: optional diagonal ridge added before pinv.
      rcond: cutoff for small singular values. If None, torch chooses a default.

    Returns:
      Lambda: (d,d) symmetric pseudo-inverse.
    """
    Sigma = symmetrize(Sigma)
    d = Sigma.shape[0]

    if shrink is not None:
        Sigma = shrink_full(Sigma, alpha=shrink)

    if ridge != 0.0:
        Sigma = Sigma + ridge * torch.eye(d, device=Sigma.device, dtype=Sigma.dtype)

    # torch.linalg.pinv uses SVD under the hood (stable but slower).
    if rcond is None:
        Lambda = torch.linalg.pinv(Sigma)
    else:
        Lambda = torch.linalg.pinv(Sigma, rcond=rcond)

    return symmetrize(Lambda)

def precision_to_partial_corr(Lambda_t: torch.Tensor, eps: float = 1e-12):
    """
    Returns:
      P_t: unit-diagonal normalized precision (partial-correlation-like)
      d_t: diag(Lambda_t) (task-specific precision diagonal)
    """
    Lambda_t = 0.5 * (Lambda_t + Lambda_t.T)
    d_t = torch.diagonal(Lambda_t).clamp_min(eps).clone()
    P_t = normalize_to_unit_diagonal(Lambda_t, eps=eps)
    return P_t, d_t

def update_shared_precision_corr_equal_tasks(
    Sigma_t: torch.Tensor,
    P_sum_old: torch.Tensor,
    num_tasks: int,
    eps: float = 1e-12,
    cov_ridge: float = 1e-6,
    shrink: float = 1e-3,
):
    """
    Equal task weighting online update in precision space.

    Inputs:
      Sigma_t: (d,d) covariance for current task
      P_sum_old: (d,d) running sum of P_t (normalized precision matrices)
      num_tasks: total number of tasks after including this task
                (same convention as your code: you pass num_tasks explicitly)
      eps: numerical eps for diagonal clamps
      cov_ridge: ridge added before inversion
      shrink: shrinkage toward I for conditioning (so Cholesky works)

    Returns:
      P_new: (d,d) shared normalized precision (unit diagonal)
      Lp: (d,d) Cholesky of regularized shared P_new (for solves if needed)
      d_t: (d,) task-specific precision diagonal (store per task)
      P_sum_new: updated sum accumulator
      P_t: current task's normalized precision
    """
    # 1) task precision
    Lambda_t = precision_from_cov_pinv(Sigma_t, ridge=cov_ridge, shrink=0.05) #cov_to_precision(Sigma_t, ridge=cov_ridge)#

    # 2) normalize precision to unit diagonal
    P_t, d_t = precision_to_partial_corr(Lambda_t, eps=eps)

    # 3) accumulate
    P_sum_new = P_sum_old + P_t

    # 4) average + normalize (unit diagonal)
    P_avg = P_sum_new / float(num_tasks)
    P_new = normalize_to_unit_diagonal(P_avg, eps=eps)

    # 5) shrink to identity for SPD / conditioning
    d = P_new.shape[0]
    P_reg = (1.0 - shrink) * P_new + shrink * torch.eye(d, device=P_new.device, dtype=P_new.dtype)

    # 6) cholesky (optional, but consistent with your structure)
    Lp = None#torch.linalg.cholesky(P_reg)

    return P_new, Lp, d_t, P_sum_new, P_t, P_reg

def mahalanobis_sq_paired_batch_from_precision_corr(
    X: torch.Tensor, Y: torch.Tensor,
    P_shared: torch.Tensor,
    d_t: torch.Tensor,
    eps_identity: float = 1e-8
):
    """
    Uses Lambda_hat = S P_shared S, with S = diag(sqrt(d_t)).
    Computes d^2 = diff^T Lambda_hat diff for paired batches.
    """
    diff = X - Y  # (B,d)
    s = torch.sqrt(d_t + eps_identity)  # (d,)

    # z = S diff  (elementwise)
    z = diff * s  # (B,d)

    # d^2 = z^T P_shared z
    v = z @ P_shared  # (B,d)
    return (v * z).sum(dim=1)  # (B,)

def mahalanobis_sq_to_tasks_from_precision_corr(
    X: torch.Tensor,          # (B, d)
    Y: torch.Tensor,          # (T, d)  task means/prototypes
    P_reg: torch.Tensor,      # (d, d)  shared normalized precision (regularized)
    d_all: torch.Tensor,      # (T, d)  task-specific precision diagonals
    eps_identity: float = 1e-8,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """
    Computes squared Mahalanobis distances from each x in X to each task mean in Y:

      d^2[b, t] = (x_b - y_t)^T (S_t P_reg S_t) (x_b - y_t)
    where S_t = diag(sqrt(d_all[t] + eps_identity)).

    Returns:
      D2: (B, T)
    """

    assert X.dim() == 2 and Y.dim() == 2 and P_reg.dim() == 2 and d_all.dim() == 2
    B, d = X.shape
    T, d2 = Y.shape
    assert d == d2 == P_reg.shape[0] == P_reg.shape[1] == d_all.shape[1]
    assert d_all.shape[0] == T

    # precompute per-task scaling vectors s_t = sqrt(d_t + eps)
    s_all = torch.sqrt(d_all + eps_identity)  # (T, d)

    if chunk_size is None:
        # diff: (B, T, d)
        diff = X[:, None, :] - Y[None, :, :]
        # z: (B, T, d)   task-specific scaling
        z = diff * s_all[None, :, :]
        # v: (B, T, d) = z @ P_reg
        v = torch.matmul(z, P_reg)  # matmul over last dim
        # d2: (B, T)
        return (v * z).sum(dim=-1)

    # Memory-friendly chunking over tasks
    D2_chunks = []
    for start in range(0, T, chunk_size):
        end = min(T, start + chunk_size)
        Yc = Y[start:end]                 # (Tc, d)
        sc = s_all[start:end]             # (Tc, d)

        diff = X[:, None, :] - Yc[None, :, :]    # (B, Tc, d)
        z = diff * sc[None, :, :]                # (B, Tc, d)
        v = torch.matmul(z, P_reg)               # (B, Tc, d)
        D2_chunks.append((v * z).sum(dim=-1))    # (B, Tc)

    return torch.cat(D2_chunks, dim=1)  # (B, T)


def mahalanobis_sq_to_tasks_sharedU(X, Y_all, U, lam_all, eps: float = 1e-8, chunk_size=None):
    B, d = X.shape
    T, _ = Y_all.shape
    inv_lam_all = 1.0 / (lam_all + eps)  # (T,k)

    if chunk_size is None:
        diff = X[:, None, :] - Y_all[None, :, :]   # (B,T,d)
        z = torch.matmul(diff, U)                  # (B,T,k)
        return (z * z * inv_lam_all[None, :, :]).sum(dim=-1)  # (B,T)

    outs = []
    for s in range(0, T, chunk_size):
        e = min(T, s + chunk_size)
        diff = X[:, None, :] - Y_all[None, s:e, :]
        z = torch.matmul(diff, U)
        outs.append((z * z * inv_lam_all[None, s:e, :]).sum(dim=-1))
    return torch.cat(outs, dim=1)

def get_task_distances(features, class_prototypes, strategy='min', take_sqrt=True):
    """
    Compute Mahalanobis-based distances from features to task prototypes.

    Args:
        features (torch.Tensor): [B, D] feature matrix of test samples.
        prototypes (dict): task_id -> tensor of shape [num_classes_in_task, D].
        covariances (dict): task_id -> tensor/list of covariance matrices [num_classes_in_task, D, D].
        strategy (str): 'min' or 'avg' — how to aggregate per-class distances per task.
        take_sqrt (bool): whether to apply sqrt to distances (default: True)
    
    Returns:
        torch.Tensor: [B, num_tasks] distances of each feature to each task.
    """
    device = features.device
    task_distances = []

    for task_id, class_protos in class_prototypes.items():
        #class_protos = class_protos
        #covs = covariances[task_id]

        dists_per_class = []
        prototypes = torch.stack([v["prototype"] for v in class_protos.values()]).to(device)  # [C, D]
        inv_covs = torch.stack([v["inv_cov"] for v in class_protos.values()]).to(device)  # [C, D, D]
        #task_proto = torch.stack([prototypes[task_dist][c]["prototype"] for c in prototypes[task_dist].keys()])

        
            #inv_cov = torch.inverse(covs[cls_idx].to(device))
        dist = mahalanobis_distance_classes(features, prototypes, inv_covs, sqrt=take_sqrt)
            #if take_sqrt:
            #    dist = dist.clamp_min(1e-9).sqrt()

        #dists_per_class = torch.stack(dists_per_class, dim=1)  # [B, num_classes]
        #print(dist)
        #print(dist.shape)
        # Aggregate per-task
        if strategy == 'min':
            task_dist,_ = torch.min(dist, dim=1)
        elif strategy == 'avg':
            task_dist = dist.mean(dim=1)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        task_distances.append(task_dist)

    task_distances = torch.cat(task_distances, dim=0)  # [B, num_tasks]
    return task_distances


# def id_to_domain_map(image_id, dataset):
#     if dataset == 'bigearthnet':
    
#     elif dataset == 'flair':

#     elif dataset == 'officehome':

    

class WeightingStats:
    def __init__(self, args, num_tasks, current_task, max_k=5):
        self.num_tasks = num_tasks
        self.max_k = max_k
        self.current_task = current_task
        self.args = args

        if self.args.get('efficient_inference',False):
            self.total_weights = torch.zeros(self.args['batch_size'],current_task+1)
            self.max_counts = torch.zeros(self.args['batch_size'],current_task+1)
        else:
            self.total_weights = torch.zeros(current_task+1)
            self.max_counts = torch.zeros(current_task+1)
        self.counts = 0
        self._all_weights = []

        # Agreement statistics
        self.rank_counts = torch.zeros(max_k, dtype=torch.long)  # how often true task is in top-k
        self.total_samples = 0
        self.task_class_confusion = {'task_class':0, 'no_task_class':0, 'task_no_class':0, 'no_task_no_class':0}
        self.additional_stats = {}
        self.compute_confusion = True
        

    def update(self, weights: torch.Tensor, correct_task=None):
        """Update statistics for a single sample.
        Args:
            weights (torch.Tensor): shape (num_tasks,)
            correct_task (int): ground truth task ID
        """
        self.total_weights += weights
        self.counts += 1
        self._all_weights.append(weights.detach())

        self.total_samples += 1

        if self.args.get('efficient_inference',False):
            # Sort tasks by weight (descending)
            ranked_tasks = torch.argsort(weights, descending=True)
            #print(ranked_tasks)
            #if ranked_tasks
            max_task = weights.argmax(dim=1)
            self.max_counts[max_task] += 1
        else:
             # Sort tasks by weight (descending)
            ranked_tasks = torch.argsort(weights, descending=True)
            #print(ranked_tasks)
            #if ranked_tasks
            max_task = weights.argmax().item()#max_task = weights.argmax(dim=1).item()
            self.max_counts[max_task] += 1

        # Check top-k matches
        if correct_task != None:
            for k in range(self.max_k):
                if correct_task in ranked_tasks[: k + 1]:
                    self.rank_counts[k] += 1

    def update_confusion(self, task_correct, class_correct):
        if task_correct and class_correct:
            self.task_class_confusion['task_class'] += 1
        elif task_correct and not class_correct:
            self.task_class_confusion['task_no_class'] += 1
        elif not task_correct and class_correct:
            self.task_class_confusion['no_task_class'] += 1
        else:
            self.task_class_confusion['no_task_no_class'] += 1

    def update_additional_stats(self, key, value):
        if key in self.additional_stats.keys():
            self.additional_stats[key].append(value)
        else:
            self.additional_stats[key] = [value]

    def compute(self):
        avg_weights = self.total_weights / max(1, self.counts)
        variances = torch.var(torch.stack(self._all_weights), dim=0) if self._all_weights else torch.zeros(self.current_task+1)
        max_freqs = self.max_counts / max(1, self.counts)

        # Relative agreement statistics
        topk_agreement = {f"top{k+1}_agreement": self.rank_counts[k].item() / max(1, self.total_samples)
                          for k in range(self.max_k)}
        
        if self.compute_confusion:
            relative_confusion = self.task_class_confusion
            for key in relative_confusion.keys():
                relative_confusion[key] = relative_confusion[key]/self.total_samples

        stats = {
            "avg_weights": avg_weights.tolist(),
            "variances": variances.tolist(),
            "topk_agreement": topk_agreement,
            "max_frequencies": max_freqs.tolist(),
        }
        if self.compute_confusion:
            stats["relative_confusion"] = relative_confusion
        
        for key in self.additional_stats.keys():
            stats[key] = {'mean': torch.mean(torch.stack(self.additional_stats[key])).item(), 'var':torch.var(torch.stack(self.additional_stats[key])).item()}
        return stats


