"""
File containing the function to load all the frame datasets.
"""

#Standard imports
import os

#Local imports
from util.dataset import load_classes
from dataset.frame import ActionSpotDataset, ActionSpotVideoDataset, ActionSpotDatasetJoint

#Constants
STRIDE = 1
STRIDE_SN = 12
STRIDE_SNB = 2
OVERLAP = 0.9
OVERLAP_SN = 0.5

def get_datasets(args):
    classes = load_classes(os.path.join('data', args.dataset, 'class.txt'))

    dataset_len = args.epoch_num_frames // args.clip_len
    stride = STRIDE
    overlap = OVERLAP
    if args.dataset == 'soccernet':
        stride = STRIDE_SN
        overlap = OVERLAP_SN
    elif args.dataset == 'soccernetball':
        stride = STRIDE_SNB

    dataset_kwargs = {
        'stride': stride, 'overlap': overlap, 'radi_displacement': args.radi_displacement,
        'mixup': args.mixup, 'dataset': args.dataset
    }

    print('Dataset size:', dataset_len)

    train_data = ActionSpotDataset(
        classes, os.path.join('data', args.dataset, 'train.json'),
        args.frame_dir, args.store_dir, args.store_mode, 
        args.modality, args.clip_len, dataset_len, **dataset_kwargs)
    train_data.print_info()
        
    dataset_kwargs['mixup'] = False # Disable mixup for validation

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
        
    #In case of using pretrain, datasets with additional data
    pretrain_classes = None
    if args.pretrain != None:

        stride_pretrain = STRIDE
        overlap_pretrain = OVERLAP
        if args.pretrain['dataset'] == 'soccernet':
            stride_pretrain = STRIDE_SNB
            overlap_pretrain = OVERLAP_SN
        elif args.dataset == 'soccernetball':
            stride_pretrain = STRIDE_SNB

        dataset_pretrain_kwargs = {
            'stride': stride_pretrain, 'overlap': overlap_pretrain, 'radi_displacement': args.radi_displacement,
            'mixup': args.mixup, 'dataset': args.pretrain['dataset']
        }

        pretrain_classes = load_classes(os.path.join('data', args.pretrain['dataset'], 'class.txt'))

        pretrain_train_data = ActionSpotDataset(
            pretrain_classes, os.path.join('data', args.pretrain['dataset'], 'train.json'),
            args.pretrain['frame_dir'], args.pretrain['store_dir'], args.store_mode,
            args.modality, args.clip_len, dataset_len, **dataset_pretrain_kwargs)
        pretrain_train_data.print_info()

        dataset_pretrain_kwargs['mixup'] = False # Disable mixup for validation

        pretrain_val_data = ActionSpotDataset(
            pretrain_classes, os.path.join('data', args.pretrain['dataset'], 'val.json'),
            args.pretrain['frame_dir'], args.pretrain['store_dir'], args.store_mode,
            args.modality, args.clip_len, dataset_len // 4, **dataset_pretrain_kwargs)
        pretrain_val_data.print_info()

        train_data = ActionSpotDatasetJoint(train_data, pretrain_train_data)
        val_data = ActionSpotDatasetJoint(val_data, pretrain_val_data)
        
    return classes, pretrain_classes, train_data, val_data, val_data_frames