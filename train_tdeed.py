#!/usr/bin/env python3
"""
File containing the main training script for T-DEED.
"""

#Standard imports
import argparse
import os
import time
import torch
import numpy as np
import random
from torch.utils.data import DataLoader


#Local imports
from util.io import load_json, store_json
from dataset.datasets import get_datasets
from model.model import TDEEDModel
from torch.optim.lr_scheduler import (
    ChainedScheduler, LinearLR, CosineAnnealingLR)
from util.eval import evaluate
from dataset.frame import ActionSpotVideoDataset


#Constants
INFERENCE_BATCH_SIZE = 4
EVAL_SPLITS = ['test']
STRIDE = 1


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')
    parser.add_argument('--seed', type=int, default=1)
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model + '-' + str(args.seed)
    args.store_dir = config['store_dir']
    args.store_mode = config['store_mode']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.crop_dim = config['crop_dim']
    args.dataset = config['dataset']
    args.radi_displacement = config['radi_displacement']
    args.epoch_num_frames = config['epoch_num_frames']
    args.feature_arch = config['feature_arch']
    args.learning_rate = config['learning_rate']
    args.mixup = config['mixup']
    args.modality = config['modality']
    args.num_classes = config['num_classes']
    args.num_epochs = config['num_epochs']
    args.warm_up_epochs = config['warm_up_epochs']
    args.start_val_epoch = config['start_val_epoch']
    args.temporal_arch = config['temporal_arch']
    args.n_layers = config['n_layers']
    args.sgp_ks = config['sgp_ks']
    args.sgp_r = config['sgp_r']
    args.only_test = config['only_test']
    args.criterion = config['criterion']
    args.num_workers = config['num_workers']

    return args

def get_lr_scheduler(args, optimizer, num_steps_per_epoch):
    cosine_epochs = args.num_epochs - args.warm_up_epochs
    print('Using Linear Warmup ({}) + Cosine Annealing LR ({})'.format(
        args.warm_up_epochs, cosine_epochs))
    return args.num_epochs, ChainedScheduler([
        LinearLR(optimizer, start_factor=0.01, end_factor=1.0,
                 total_iters=args.warm_up_epochs * num_steps_per_epoch),
        CosineAnnealingLR(optimizer,
            num_steps_per_epoch * cosine_epochs)])


def main(args):
    #Set seed
    initial_time = time.time()
    print('Setting seed to: ', args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    config_path = args.model.split('_')[0] + '/' + args.model + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    assert args.batch_size % args.acc_grad_iter == 0
    if args.crop_dim <= 0:
        args.crop_dim = None

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes, train_data, val_data, val_data_frames = get_datasets(args)
        
    if args.store_mode == 'store':
        print('Datasets have correctly been stored!')
    else:
        print('Datasets have been loaded from previous versions correctly!')


    def worker_init_fn(id):
        random.seed(id + epoch * 100)
    loader_batch_size = args.batch_size // args.acc_grad_iter


    # Dataloaders
    train_loader = DataLoader(
        train_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=2, worker_init_fn=worker_init_fn)
        
    val_loader = DataLoader(
        val_data, shuffle=False, batch_size=loader_batch_size,
        pin_memory=True, num_workers=args.num_workers,
        prefetch_factor=2, worker_init_fn=worker_init_fn)
                
    # Model
    model = TDEEDModel(args=args)
    optimizer, scaler = model.get_optimizer({'lr': args.learning_rate})

    if not args.only_test:
        # Warmup schedule
        num_steps_per_epoch = len(train_loader) // args.acc_grad_iter
        num_epochs, lr_scheduler = get_lr_scheduler(
            args, optimizer, num_steps_per_epoch)
        
        losses = []
        best_criterion = 0 if args.criterion == 'map' else float('inf')
        epoch = 0

        print('START TRAINING EPOCHS')
        for epoch in range(epoch, num_epochs):

            time_train0 = time.time()
            train_loss = model.epoch(
                train_loader, optimizer, scaler,
                lr_scheduler=lr_scheduler, acc_grad_iter=args.acc_grad_iter)
            time_train1 = time.time()
            time_train = time_train1 - time_train0
            
            time_val0 = time.time()
            val_loss = model.epoch(val_loader, acc_grad_iter=args.acc_grad_iter)
            time_val1 = time.time()
            time_val = time_val1 - time_val0

            better = False
            val_mAP = 0
            if args.criterion == 'loss':
                if val_loss < best_criterion:
                    best_criterion = val_loss
                    better = True
            elif args.criterion == 'map':
                if epoch >= args.start_val_epoch:
                    time_map0 = time.time()
                    val_mAP = evaluate(model, val_data_frames, 'VAL', classes,
                                        printed=False, test=False)
                    time_map1 = time.time()
                    time_map = time_map1 - time_map0
                    if val_mAP > best_criterion:
                        best_criterion = val_mAP
                        better = True
            
            #Printing info epoch
            print('[Epoch {}] Train loss: {:0.5f} Val loss: {:0.5f}'.format(
                epoch, train_loss, val_loss))
            if (args.criterion == 'map') & (epoch >= args.start_val_epoch):
                print('Val mAP: {:0.5f}'.format(val_mAP))
                if better:
                    print('New best mAP epoch!')
            print('Time train: ' + str(int(time_train // 60)) + 'min ' + str(np.round(time_train % 60, 2)) + 'sec')
            print('Time val: ' + str(int(time_val // 60)) + 'min ' + str(np.round(time_val % 60, 2)) + 'sec')
            if (args.criterion == 'map') & (epoch >= args.start_val_epoch):
                print('Time map: ' + str(int(time_map // 60)) + 'min ' + str(np.round(time_map % 60, 2)) + 'sec')

            losses.append({
                'epoch': epoch, 'train': train_loss, 'val': val_loss,
                'val_mAP': val_mAP
            })

            if args.save_dir is not None:
                os.makedirs(args.save_dir, exist_ok=True)
                store_json(os.path.join(args.save_dir, 'loss.json'), losses,
                            pretty=True)

                if better:
                    torch.save(
                        model.state_dict(),
                        os.path.join(os.getcwd(), 'checkpoints', args.model.split('_')[0], args.model, 'checkpoint_best.pt'))

    print('START INFERENCE')
    model.load(torch.load(os.path.join(
        os.getcwd(), 'checkpoints', args.model.split('_')[0], args.model, 'checkpoint_best.pt')
    ))

    eval_splits = EVAL_SPLITS

    for split in eval_splits:
        split_path = os.path.join(
            'data', args.dataset, '{}.json'.format(split))

        stride = STRIDE

        if os.path.exists(split_path):
            split_data = ActionSpotVideoDataset(
                classes, split_path, args.frame_dir, args.modality,
                args.clip_len, overlap_len = args.clip_len // 4 * 3, #75% overlapping
                stride = stride, dataset = args.dataset)

            pred_file = None
            if args.save_dir is not None:
                pred_file = os.path.join(
                    args.save_dir, 'pred-{}'.format(split))

            evaluate(model, split_data, split.upper(), classes, pred_file, printed = True, 
                    test = True, augment = False)
    
    print('CORRECTLY FINISHED TRAINING AND INFERENCE')




if __name__ == '__main__':
    main(get_args())