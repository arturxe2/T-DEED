#!/usr/bin/env python3
"""
File containing the inference script for T-DEED.
"""

#Standard imports
import argparse
import os
import time
import torch
import numpy as np
import random
from torch.utils.data import DataLoader
import wandb
import sys


#Local imports
from util.io import load_json, store_json
from model.model import TDEEDModel
from util.eval import inference
from dataset.frame import ActionSpotInferenceDataset
from util.dataset import load_classes


#Constants
EVAL_SPLITS = ['test']
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2


def get_args():
    #Basic arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--video_path', type=str, help='Path to video file for inference', required=True)
    parser.add_argument('--frame_width', type=int, default=796, help='Frame width for inference')
    parser.add_argument('--frame_height', type=int, default=448, help='Frame height for inference')
    parser.add_argument('--inference_threshold', type=float, default=0.2, help='Threshold for inference')
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')
    parser.add_argument('--seed', type=int, default=1)
    
    return parser.parse_args()

def update_args(args, config):
    #Update arguments with config file
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
    args.store_dir = config['store_dir']
    args.store_mode = config['store_mode']
    args.batch_size = config['batch_size']
    args.clip_len = config['clip_len']
    args.crop_dim = config['crop_dim']
    if args.crop_dim <= 0:
        args.crop_dim = None
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
    if 'pretrain' in config:
        args.pretrain = config['pretrain']
    else:
        args.pretrain = None

    return args


def main(args):

    config_path = args.model.split('_')[0] + '/' + args.model + '.json'
    config = load_json(os.path.join('config', config_path))
    args = update_args(args, config)

    # Get datasets train, validation (and validation for map -> Video dataset)
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))
    if args.pretrain != None:
        pretrain_classes = load_classes(os.path.join('data', args.pretrain['dataset'], 'class.txt'))

                
    # Model
    model = TDEEDModel(args=args)

    #If pretrain -> 2 prediction heads
    if args.pretrain != None:
        n_classes = [len(classes)+1, len(pretrain_classes)+1]
        model._model.update_pred_head(n_classes)
        model._num_classes = np.array(n_classes).sum() 

    print('START INFERENCE')
    model.load(torch.load(os.path.join(
        os.getcwd(), 'checkpoints', args.model.split('_')[0], args.model, 'checkpoint_best.pt')))

    stride = STRIDE
    if args.dataset == 'soccernet':
        stride = STRIDE_SN
    if args.dataset == 'soccernetball':
        stride = STRIDE_SNB

    inference_dataset = ActionSpotInferenceDataset(
        args.video_path, clip_len = args.clip_len, overlap_len = args.clip_len // 4 * 3 if args.dataset != 'soccernet' else args.clip_len // 2,
        stride = stride, dataset = args.dataset, size = (args.frame_width, args.frame_height)
    )

    inference_loader = DataLoader(
        inference_dataset, batch_size = args.batch_size,
        shuffle = False, num_workers = args.num_workers,
        pin_memory = True, drop_last = False)

    inference(model, inference_loader, classes, threshold = args.inference_threshold)
    
    print('CORRECTLY FINISHED INFERENCE STEP')


if __name__ == '__main__':
    main(get_args())