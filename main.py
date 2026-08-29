from email import parser
import json
import argparse
from trainer import train


def main():
    args = setup_parser().parse_args()
    param = load_json(args.config)
    args = vars(args) # Converting argparse Namespace to a dict.
    
    #print('seed', args["seed"])

    # automatically override matching keys
    for key, value in args.items():
        if key == "config":
            continue

        if value is not None:
            param[key] = value

    args.update(param) # Add parameters from json
    train(args)
    

def load_json(setting_path):
    with open(setting_path) as data_file:
        param = json.load(data_file)
    return param

def setup_parser():
    parser = argparse.ArgumentParser(description='Reproduce of multiple pre-trained incremental learning algorthms.')
    #parser.add_argument('--config', type=str, default='./exps/simplecil.json', help='Json file of settings.')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument('--seed', type=int, nargs="+", help='The seed value')
    # parser.add_argument("--local_rank", type=int, default=0)
    
    # optional overrides
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--merging", type=str, default=None)
    parser.add_argument("--distance", type=str, default=None)
    parser.add_argument("--backbone_type", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=None) 
    parser.add_argument("--shrinkage", type=float, default=None)
    parser.add_argument("--task_weighting_strat", type=str, default=None)
    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--init_epoch", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lora_rank", type=int, default=None)
    parser.add_argument("--init_lr", type=float, default=None)
    parser.add_argument("--lrate", type=float, default=None)
    #parser.add_argument("--merged_head", type=bool, default=None)
    #parser = argparse.ArgumentParser()

    return parser

if __name__ == '__main__':
    main()
