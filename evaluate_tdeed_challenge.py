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
import wandb
import sys


#Local imports
from util.io import load_json, load_text
from util.dataset import load_classes
from model.model import TDEEDModel
from util.eval import evaluate, evaluate_SNB
from SoccerNet.Evaluation.ActionSpotting import evaluate as evaluate_SN
from dataset.frame import ActionSpotVideoDataset
from train_tdeed import get_args, update_args


#Constants
EVAL_SPLITS = ['challenge']
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2


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

    assert args.dataset in ['soccernetball'] # Only SoccerNet Ball is supported

    #Variables for SN & SNB label paths if datastes
    if (args.dataset == 'soccernet') | (args.dataset == 'soccernetball'):
        global LABELS_SN_PATH
        global LABELS_SNB_PATH
        LABELS_SN_PATH = load_text(os.path.join('data', 'soccernet', 'labels_path.txt'))[0]
        LABELS_SNB_PATH = load_text(os.path.join('data', 'soccernetball', 'labels_path.txt'))[0]

    assert args.batch_size % args.acc_grad_iter == 0
    if args.crop_dim <= 0:
        args.crop_dim = None

    # initialize wandb
    wandb.login()
    wandb.init(config = args, dir = args.save_dir + '/wandb_logs', project = 'ExtendTDEED', name = args.model + '-' + str(args.seed))
                
    # Model
    model = TDEEDModel(args=args)

    #If pretrain -> 2 prediction heads
    if args.pretrain != None:
        classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))
        pretrain_classes = load_classes(os.path.join('data', args.pretrain['dataset'], 'class.txt'))
        n_classes = [len(classes)+1, len(pretrain_classes)+1]
        model._model.update_pred_head(n_classes)
        model._num_classes = np.array(n_classes).sum() 

    print('START INFERENCE')
    model.load(torch.load(os.path.join(
        os.getcwd(), 'checkpoints', args.model.split('_')[0], args.model, 'checkpoint_best.pt')))

    eval_splits = EVAL_SPLITS

    for split in eval_splits:
        split_path = os.path.join(
            'data', args.dataset, '{}.json'.format(split))

        stride = STRIDE
        if args.dataset == 'soccernet':
            stride = STRIDE_SN
        if args.dataset == 'soccernetball':
            stride = STRIDE_SNB

        if os.path.exists(split_path):
            split_data = ActionSpotVideoDataset(
                classes, split_path, args.frame_dir, args.modality,
                args.clip_len, overlap_len = args.clip_len // 4 * 3 if args.dataset != 'soccernet' else args.clip_len // 2, # 3/4 overlap for video dataset, 1/2 overlap for soccernet
                stride = stride, dataset = args.dataset)

            pred_file = None
            if args.save_dir is not None:
                pred_file = os.path.join(
                    args.save_dir, 'pred-{}'.format(split))

            mAPs, tolerances = evaluate(model, split_data, split.upper(), classes, pred_file, printed = True, 
                        test = True, augment = (args.dataset != 'soccernet') & (args.dataset != 'soccernetball'))
            
            if split != 'challenge':
                for i in range(len(mAPs)):
                    wandb.log({'test/mAP@' + str(tolerances[i]): mAPs[i]})
                    wandb.summary['test/mAP@' + str(tolerances[i])] = mAPs[i]

                if args.dataset == 'soccernet':
                    results = evaluate_SN(LABELS_SN_PATH, '/'.join(pred_file.split('/')[:-1]) + '/preds', 
                                split = split, prediction_file = "results_spotting.json", version = 2, 
                                metric = "tight")

                    print('Tight aMAP: ', results['a_mAP'] * 100)
                    print('Tight aMAP per class: ', results['a_mAP_per_class'])

                    wandb.log({'test/mAP': results['a_mAP'] * 100})
                    wandb.summary['test/mAP'] = results['a_mAP'] * 100

                    for j in range(len(classes)):
                        wandb.log({'test/classes/mAP@' + list(classes.keys())[j]: results['a_mAP_per_class'][j] * 100})

                if args.dataset == 'soccernetball':
                    results = evaluate_SNB(LABELS_SNB_PATH, '/'.join(pred_file.split('/')[:-1]) + '/preds', split = split)
                    
                    print('aMAP@1: ', results['a_mAP'] * 100)
                    print('Average mAP per class: ')
                    print('-----------------------------------')
                    for i in range(len(results["a_mAP_per_class"])):
                        print("    " + list(classes.keys())[i] + ": " + str(np.round(results["a_mAP_per_class"][i] * 100, 2)))

                    wandb.log({'test/mAP@1': results['a_mAP'] * 100})
                    wandb.summary['test/mAP@1'] = results['a_mAP'] * 100

                    for j in range(len(classes)):
                        wandb.log({'test/classes/mAP@' + list(classes.keys())[j]: results['a_mAP_per_class'][j] * 100})


    
    print('CORRECTLY FINISHED INFERENCE')




if __name__ == '__main__':
    main(get_args())