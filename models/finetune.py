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
from models.base import BaseLearner
from utils.toolkit import target2onehot, tensor2numpy
from utils.toolkit import count_parameters

import timm
from backbone.lora_finetune import LoRA_ViT_timm
import torch.distributed as dist
from sklearn.metrics import average_precision_score

import os

num_workers = 8

class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self._network = IncrementalNet(args, True)

    def after_task(self):
        self._known_classes = self._total_classes

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )
        if self.args.get("multi-label",False):
            self._total_classes = 19
        
        self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        # logging.info(
        #     "Trainable params: {}".format(count_parameters(self._network, True))
        # )

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
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
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
        if self.args.get('multi_head',False):
            self._total_classes = self._known_classes + data_manager.nb_classes#19
        else: self._total_classes = data_manager.nb_classes
        
        if self._cur_task == 0 or self.args.get('multi_head',False):
            self._network.update_fc(self._total_classes)
        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )
        #
        print(data_manager._class_order[self._cur_task])
        if self.args.get('joined',False):
            train_domains = data_manager._class_order[:self._cur_task+1]
        else:
            train_domains = [data_manager._class_order[self._cur_task]]


        if self.args['dataset']!='bigearthnet':
            train_dataset = data_manager.get_dataset_ml(
                domains=train_domains,
                source="train",
                mode="train",
            )
            print('train samples', len(train_dataset))
            # self.train_loader = DataLoader(
            #     train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=2
            # )
            test_dataset = data_manager.get_dataset_ml(
                data_manager._class_order[:self._cur_task+1], source="test", mode="test"
            )
            print('test samples', len(test_dataset))
        else:
            train_dataset = data_manager.get_dataset_ben(
                domains=train_domains,
                source="train",
                mode="train",
            )
            print('train samples', len(train_dataset))
            
            test_dataset = data_manager.get_dataset_ben(
                data_manager._class_order[:self._cur_task+1], source="test", mode="test"
            )
            print('test samples', len(test_dataset))


        self.train_loader = DataLoader(
            train_dataset, batch_size=self.args["batch_size"], shuffle=True, num_workers=num_workers
        )

        self.test_loader = DataLoader(
            test_dataset, batch_size=self.args["batch_size"], shuffle=False, num_workers=num_workers
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
        # if use VIT-B-16
        model = timm.create_model("vit_base_patch16_224",pretrained=True, num_classes=0)

        # if use DINO
        # model = timm.create_model('vit_base_patch16_224_dino', pretrained=True, num_classes=0)

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
        rank=10
        model = LoRA_ViT_timm(vit_model=model.eval(), r=rank, num_classes=10, index=index, increment= self.args['increment'], filepath=self.args['filepath'] + self.args['prefix'] + '/', 
        cur_task_index= self._cur_task, reinit = self.args.get('joined',False), use_lora = self.args.get('use_lora',True), update_backbone = self.args.get('update_backbone',False))
        model.out_dim = 768
        return model

    def _train(self, train_loader, test_loader):
        self._network.to(self._device)
        if self._cur_task == 0:
            optimizer = self.get_optimizer()
            scheduler = self.get_scheduler(optimizer)
            # optimizer = optim.SGD(
            #     self._network.parameters(),
            #     momentum=0.9,
            #     lr=self.args["init_lr"],
            #     # weight_decay=self.args["init_weight_decay"],
            # )
            # scheduler = optim.lr_scheduler.MultiStepLR(
            #     optimizer=optimizer, milestones=self.args["init_milestones"], gamma=self.args["init_lr_decay"]
            # )
            self._init_train(train_loader, test_loader, optimizer, scheduler)

        else:
            if len(self._multiple_gpus) > 1:
                self._network = self._network.module
            if not self.args.get('update_backbone',False):  ### update_backbone means finetuning the model, therefore no reinitialization of model
                self._network.backbone = self.update_network(index=False)
            if len(self._multiple_gpus) > 1:
                self._network = nn.DataParallel(self._network, self._multiple_gpus)       
            self._network.to(self._device) 

            optimizer = self.get_optimizer()
            scheduler = self.get_scheduler(optimizer)
            # optimizer = optim.SGD(
            #     self._network.parameters(),
            #     lr=self.args["lrate"],
            #     momentum=0.9,
            # )  # 1e-5
            # scheduler = optim.lr_scheduler.MultiStepLR(
            #     optimizer=optimizer, milestones=self.args["milestones"], gamma=self.args["lrate_decay"]
            # )
            self._update_representation(train_loader, test_loader, optimizer, scheduler)

        save_lora_name = self.args['filepath']  + self.args['prefix'] + '/'

        if len(self._multiple_gpus) > 1:
            self._network.module.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.module.save_fc(save_lora_name, self._cur_task)
        else:
            self._network.backbone.save_lora_parameters(save_lora_name, self._cur_task)
            self._network.save_fc(save_lora_name, self._cur_task)

    def get_optimizer(self):
        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                filter(lambda p: p.requires_grad, self._network.parameters()), 
                momentum=0.9, 
                lr=self.init_lr,
                weight_decay=self.weight_decay
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                # lr=self.init_lr, 
                self.args["lrate"],
                # weight_decay=self.weight_decay
                betas=(0.9, 0.999)
            )
            
        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                filter(lambda p: p.requires_grad, self._network.parameters()),
                lr=self.init_lr, 
                weight_decay=self.weight_decay
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
                    inputs, targets = inputs.to(self._device), targets.to(self._device)#.to(torch.float32)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    logits = self._network(inputs)["logits"]
                    #targets = targets
                    #print(logits.shape)
                    #print(targets.shape)
                    #print(targets)
                    #print(logits.dtype)
                    #print(targets.dtype)

                    loss = F.cross_entropy(logits, targets)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    torch.cuda.synchronize()
                    elapsed += time.time() - t0
                    losses += loss.item()

                    
                    _, preds = torch.max(logits.detach(), dim=1)
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
                self.total_training_time += elapsed
                #train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
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
                for i, batch in enumerate(train_loader):
                    if self.args['dataset'] != 'bigearthnet':
                        (_, inputs, targets) = batch[0],batch[1],batch[2]
                    else:
                        (inputs, targets, _) = batch[0],batch[1],batch[2]
                    inputs, targets = inputs.to(self._device), targets.to(self._device)
                    # logits = self._network(inputs)["logits"]
                    torch.cuda.synchronize()
                    t0 = time.time()
                    logits, ortho_loss = self._network(inputs, ortho_loss=True)
                    logits = logits['logits'] 
                    

                    if not self.args.get("domain_incremental_learning", False):
                        fake_targets = targets - self._known_classes
                    else:
                        fake_targets = targets 
                    
                    #print(fake_targets) 
                    #print(fake_targets.dtype) 
                    # loss_clf = F.cross_entropy(
                    #     logits[:, self._known_classes :], fake_targets
                    # )
                    loss_clf = F.cross_entropy(
                         logits, fake_targets
                    )

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
                    if self.args.get('multi_head',False):
                        preds = preds % (self._total_classes - self._known_classes)
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
                self.total_training_time += elapsed
                #train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
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

