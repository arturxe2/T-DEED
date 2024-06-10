"""
File containing the function to load all the frame datasets.
"""

#Standard imports
import os

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset

#Constants
STRIDE = 1

def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = STRIDE

    dataset_kwargs = {
        'stride': stride, 'overlap': 0.9, 'radi_displacement': args.radi_displacement,
        'mixup': args.mixup, 'dataset': args.dataset
    }

    print('Dataset size:', dataset_len)

    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.store_dir, args.store_mode, 
        args.modality, args.clip_len, dataset_len, **dataset_kwargs)
    train_data.print_info()
        

    val_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'val.json'),
        args.frame_dir, args.store_dir, args.store_mode,
        args.modality, args.clip_len, dataset_len // 4, **dataset_kwargs)
    val_data.print_info()

    val_data_frames = None
    if args.criterion == 'map':
        # Only perform mAP evaluation during training if criterion is mAP
        val_data_frames = ActionSpotVideoDataset(
            classes, os.path.join('data', args.dataset, 'val.json'),
            args.frame_dir, args.modality, args.clip_len,
            overlap_len=0, stride = stride, dataset = args.dataset)        
    
    return classes, train_data, val_data, val_data_frames