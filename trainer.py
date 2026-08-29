import sys
import logging
import copy
import time
import torch
from utils import factory
from utils.data_manager_dil import DataManager as DataManagerDIL
from utils.toolkit import count_parameters
import os
import numpy as np
import shutil


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )

    if os.path.exists(logfilename + ".log" ):
        for i in range(100):
            if not os.path.exists(logfilename + '_{}'.format(i) + ".log"):
                logfilename = logfilename + '_{}'.format(i)
                break

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    if args['dataset'] == 'flair':
        data_manager = DataManagerDIL(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )
    
    elif args['dataset'] == 'office_home':
        data_manager = DataManagerDIL(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )
    elif args['dataset'] == 'domainnet':
        data_manager = DataManagerDIL(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )
    elif args['dataset'] == 'bigearthnet':
        data_manager = DataManagerDIL(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )
    elif args['dataset'] == 'core50':
        data_manager = DataManagerDIL(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )
    else:     
        data_manager = DataManager(
            args["dataset"],
            args["shuffle"],
            args["seed"],
            args["init_cls"],
            args["increment"],
            args,
        )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    if args.get('multi_head',False) and (args["model_name"] in ["l2p","dualprompt","coda_prompt"]):
        args["nb_classes"] = data_manager.nb_classes * data_manager.nb_tasks
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    print("Number of Classes: ", args["nb_classes"])
    print("Number of Tasks: ", args["nb_tasks"])
    cnn_curve, nme_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix, nme_matrix = [], []


    total_train_phase_time = 0
    total_test_phase_time = 0

    raw_training_time_old = 0
    raw_testing_time_old = 0
    for task in range(data_manager.nb_tasks):
        # task = 9
        print('task',task)
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )
        
        torch.cuda.synchronize()
        t0 = time.time()
        if not args.get('domain_incremental_learning', False):
            model.incremental_train(data_manager)
        else:
            model.incremental_train_dil(data_manager)
        if args["model_name"] == "ttime" or args["model_name"] == "ttime-peft":
            #model.compute_class_prototypes_and_covariances(model.train_loader)
            if not args.get('class_based_prototypes',False):
                model.compute_task_prototype_and_covariance(model.train_loader, shrinkage=args.get('shrinkage', 0.05))
            else:
                model.compute_task_prototype_and_covariance_class_based(model.train_loader, shrinkage=args.get('shrinkage', 0.05))
            #model.backbone = model.update_network(index=False)
            #model.compute_task_prototype_within_class_cov(model.train_loader)
            #model.compute_task_similarity_matrix(task_means=model.task_prototypes, task_covs=model.task_covariances)
        torch.cuda.synchronize()
        t1 = time.time()
        elapsed_train = t1-t0
        total_train_phase_time += elapsed_train

        cnn_accy, nme_accy = model.eval_task()
        torch.cuda.synchronize()
        elapsed_test = time.time() - t1
        total_test_phase_time += elapsed_test
        model.after_task()

        if nme_accy is not None:
            logging.info("CNN: {}".format(cnn_accy["grouped"]))
            logging.info("NME: {}".format(nme_accy["grouped"]))

            cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
            cnn_keys_sorted = sorted(cnn_keys)
            cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
            cnn_matrix.append(cnn_values)

            nme_keys = [key for key in nme_accy["grouped"].keys() if '-' in key]
            nme_keys_sorted = sorted(nme_keys)
            nme_values = [nme_accy["grouped"][key] for key in nme_keys_sorted]
            nme_matrix.append(nme_values)

            cnn_curve["top1"].append(cnn_accy["top1"])
            cnn_curve["top5"].append(cnn_accy["top5"])

            nme_curve["top1"].append(nme_accy["top1"])
            nme_curve["top5"].append(nme_accy["top5"])

            logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
            logging.info("CNN top5 curve: {}".format(cnn_curve["top5"]))
            logging.info("NME top1 curve: {}".format(nme_curve["top1"]))
            logging.info("NME top5 curve: {}\n".format(nme_curve["top5"]))

            print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
            print('Average Accuracy (NME):', sum(nme_curve["top1"])/len(nme_curve["top1"]))

            logging.info("Average Accuracy (CNN): {}".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
            logging.info("Average Accuracy (NME): {}".format(sum(nme_curve["top1"])/len(nme_curve["top1"])))

        else:
            if not args.get('multi-label',False): 
                logging.info("No NME accuracy.")
                logging.info("CNN: {}".format(cnn_accy["grouped"]))

                cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
                cnn_keys_sorted = sorted(cnn_keys)
                cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
                #cnn_matrix.append(cnn_values)

                cnn_curve["top1"].append(cnn_accy["top1"])
                cnn_curve["top5"].append(cnn_accy["top5"])

                logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

                print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
                logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))

            else:
                logging.info("No NME accuracy.")
                #logging.info("CNN: {}".format(cnn_accy["grouped"]))

                #cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
                #cnn_keys_sorted = sorted(cnn_keys)
                #cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys_sorted]
                #cnn_matrix.append(cnn_values)

                cnn_curve["top1"].append(cnn_accy["top1"])
                #cnn_curve["top5"].append(cnn_accy["top5"])

                logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
                #logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

                print('Average Accuracy (CNN):', sum(cnn_curve["top1"])/len(cnn_curve["top1"]))
                logging.info("Average Accuracy (CNN): {} \n".format(sum(cnn_curve["top1"])/len(cnn_curve["top1"])))
        logging.info("Task Raw Training Time: {}".format(model.total_training_time-raw_training_time_old))
        logging.info("Task Raw Testing Time: {}".format(model.total_testing_time-raw_testing_time_old))
        raw_training_time_old = model.total_training_time
        raw_testing_time_old = model.total_testing_time    
        logging.info("Task Training Phase Time: {}".format(elapsed_train))
        logging.info("Task Testing Phase Time: {}".format(elapsed_test))
    logging.info("Total Raw Training Time: {}".format(model.total_training_time))
    logging.info("Total Raw Testing Time: {}".format(model.total_testing_time))      
    logging.info("Total Training Phase Time: {}".format(total_train_phase_time))
    logging.info("Total Testing Phase Time: {}".format(total_test_phase_time))
    if args["model_name"] == "ttime" and args.get('weighting_stats',True):
        filename = './logs/ttime/' + args['dataset'] + '/' + args["prefix"] + '_' + str(args['seed']) + '_tt_merging_stats.json'
        save_json(model.tt_stats, filename)
        fc_stats = model.compute_weight_stats()
        filename = './logs/ttime/' + args['dataset'] + '/' + args["prefix"] + '_' + str(args['seed']) + '_fc_stats.json'
        save_json(fc_stats, filename)
    if not args.get("save_model_parameters",True):
        path = args['filepath'] + args['prefix']

        if os.path.exists(path):
            shutil.rmtree(path)
        #shutil.rmtree(args['filepath'] + args['prefix'])

    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (CNN):')
        print(np_acctable)
        logging.info('Forgetting (CNN): {}'.format(forgetting))

    if len(nme_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(nme_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        np_acctable = np_acctable.T
        forgetting = np.mean((np.max(np_acctable, axis=1) - np_acctable[:, task])[:task])
        print('Accuracy Matrix (NME):')
        print(np_acctable)
        logging.info('Forgetting (NME): {}'.format(forgetting))

def _set_device(args):
    device_type = args["device"]
    gpus = []
    print('devices_type', device_type)
    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))


import json, os

def save_json(data, filename):
    base, ext = os.path.splitext(filename or "data.json")
    ext = ext or ".json"
    i, new_name = 0, filename

    while os.path.exists(new_name := f"{base}{'_' + str(i) if i else ''}{ext}"):
        i += 1

    with open(new_name, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"Saved as: {new_name}")