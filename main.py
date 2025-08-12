import os
import time
import wandb
import random
import argparse
from datetime import datetime

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

import model, trainer, utils


def get_args():
    parser = argparse.ArgumentParser()
    ### Data
    parser.add_argument('--data_path', default='./data/Amyloid', help='Type of dataset, choose Amyloid or Tau') 
    
    ### Train
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', type=int, default=0, help='Which gpu to use')
    parser.add_argument('--is_save', type=bool, default=False, help='save best model')

    ### Hyperparameter
    parser.add_argument('--is_clamp', type=bool, default=False, help='True: clamp eps>=0 / False: abs(eps)')
    parser.add_argument('--adv_w', type=float, default=0.1, help='Weight of regularizer')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--lr_lambda', type=float, default=0.995, help='Lambda of learning rate scheduler')
    parser.add_argument('--batch', type=int, default=8, help='Batch size')
    parser.add_argument('--epoch', type=int, default=1000, help='Epoch')
    parser.add_argument('--eps_init', type=float, default=0.01, help='Init of adv attack magnitude')
    parser.add_argument('--hidden_dim', type=int, default=32, help='Hidden dimension')
    parser.add_argument('--classifier', default='FT', help='MLP/FT/NODE') 
    parser.add_argument('--optim', default='SGD', help='either SGD or Adam') 

    args = parser.parse_args()
    return args

def print_args(args):
    """Print all argument values in a formatted way."""
    print("\n=== Experiment Configuration ===")
    for key, value in vars(args).items():
        print(f"{key}: {value}")
    print("==============================\n")

if __name__ == "__main__":
    args = get_args()
    print_args(args)
    random.seed(args.seed)  
    np.random.seed(args.seed)  
    torch.manual_seed(args.seed) 
    torch.cuda.manual_seed(args.seed)
    device = torch.device('cuda:' + str(args.device))
    current_time = time.strftime('%B%d_%H_%M_%S', time.localtime(time.time()))
    print('current_time: ', current_time)

    if args.data_path == './data/FDG':
        data = 'FDG'
    else: 
        data = 'Amyloid'
    wandb.init(project=data)

    if args.is_clamp == True:
        wandb.run.name = current_time
    else:
        wandb.run.name = 'abs_' + current_time
    wandb.run.save()
    wandb.config.update(args)

    # Load Data
    print('device: ', device)
    train_dataset, test_dataset, min_age, max_age = utils.load_dataset(args.data_path, device)
    train_sampler = utils.StratifiedBatchSampler(train_dataset, args.batch)
    train_loader = DataLoader(dataset=train_dataset, batch_sampler=train_sampler)
    test_loader = DataLoader(dataset=test_dataset, batch_size=len(test_dataset), shuffle=False)
    print('Loaded ', args.data_path)
    
    network = model.Network(args.hidden_dim, args.classifier, device, min_age, max_age, args.eps_init, args.is_clamp)
    total_params = sum(p.numel() for p in network.parameters())
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print('Total: ', total_params, 'Trainable: ', trainable_params)

    trainer = trainer.Trainer(
        device = device,
        network = network,
        train_loader = train_loader,
        test_loader = test_loader,
        adv_w = args.adv_w,
        lr_lambda = args.lr_lambda,
        lr = args.lr,
        batch_size = args.batch,
        epoch = args.epoch,
        optim = args.optim,
        classifier_type = args.classifier,
        data = data,
        is_save = args.is_save
    )

    start = time.time()
    trainer.train() 
    end = time.time()
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training Time {:0>2}h {:0>2}m {:05.2f}s".format(int(hours), int(minutes), seconds))
