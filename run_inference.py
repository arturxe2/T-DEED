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


#Constants
EVAL_SPLITS = ['test']
STRIDE_SN = 12
def update_args(args, config):
    args.frame_dir = config['frame_dir']
    args.save_dir = config['save_dir'] + '/' + args.model # + '-' + str(args.seed) -> in case multiple seeds
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
    if 'pretrain' in config:
        args.pretrain = config['pretrain']
    else:
        args.pretrain = None

    return args

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


    LABELS_SN_PATH = load_text(os.path.join('data', args.dataset, 'labels_path.txt'))[0]
    global LABELS_SNB_PATH

    assert args.batch_size % args.acc_grad_iter == 0
    if args.crop_dim <= 0:
        args.crop_dim = None

    # initialize wandb
    wandb.login()
    wandb.init(config = args, dir = args.save_dir + '/wandb_logs', project = 'ExtendTDEED', name = args.model + '-' + str(args.seed))
    
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    # Model
    model = TDEEDModel(args=args)


    print('START INFERENCE')
    model.load(torch.load(os.path.join(
        os.getcwd(), 'checkpoints', args.model.split('_')[0], args.model, 'checkpoint_best.pt')))

    eval_splits = EVAL_SPLITS

    for split in eval_splits:
        split_path = os.path.join(
            'data', args.dataset, '{}.json'.format(split))

        stride = STRIDE_SN

        if os.path.exists(split_path):
            split_data = ActionSpotVideoDataset(
                classes, split_path, args.frame_dir, args.modality,
                args.clip_len, overlap_len = args.clip_len // 2,
                stride = stride, dataset = args.dataset)

            # Add these debug prints
            print("\nDebug: Dataset Details")
            print(f"Clip length: {args.clip_len}")
            print(f"Modality: {args.modality}")
            print(f"Dataset: {args.dataset}")
            
            # Debug first sample frame loading
            sample = split_data[0]
            video_path = os.path.join(args.frame_dir, sample['video'])
            print("\nDebug: Frame Loading")
            frame_files = sorted(os.listdir(video_path))
            print(f"Number of frames in directory: {len(frame_files)}")
            print(f"First few frames: {frame_files[:5]}")
            print(f"Last few frames: {frame_files[-5:]}")
            print(f"Sample start frame: {sample['start']}")
            print(f"Sample frame index: {sample['frame']}")
            
            # Debug frame number extraction
            frame_numbers = [int(f.replace('frame', '').replace('.jpg', '')) for f in frame_files[:5]]
            print(f"Frame numbers: {frame_numbers}")
            
            pred_file = None
            if args.save_dir is not None:
                pred_file = os.path.join(
                    args.save_dir, 'pred-{}'.format(split))

            print("Debug: First batch shape check")
            first_batch = next(iter(DataLoader(split_data, batch_size=1)))
            print("Clip shape:", first_batch['frame'].shape)
            print("Labels shape:", first_batch['label'].shape if 'label' in first_batch else "No labels")

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
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('-ag', '--acc_grad_iter', type=int, default=1,
                        help='Use gradient accumulation')
    parser.add_argument('--seed', type=int, default=1)
    args = parser.parse_args()

    main(args)