#!/usr/bin/env python3

"""
File containing classes related to the frame datasets.
"""

#Standard imports
from util.io import load_json
import os
import random
import numpy as np
import copy
import torch
from torch.utils.data import Dataset
import torchvision
from tqdm import tqdm
import pickle

#Local imports


#Constants

# Pad the start/end of videos with empty frames
DEFAULT_PAD_LEN = 5


class ActionSpotDataset(Dataset):

    def __init__(
            self,
            classes,                    # dict of class names to idx
            label_file,                 # path to label json
            frame_dir,                  # path to frames
            store_dir,                  # path to store files (with frames path and labels per clip)
            store_mode,                 # 'store' or 'load'
            modality,                   # [rgb, bw, flow]
            clip_len,                   # Number of frames per clip
            dataset_len,                # Number of clips
            stride=1,                   # Downsample frame rate
            overlap=1,                  # Overlap between clips (in proportion to clip_len)
            radi_displacement=0,        # Radius of displacement for labels
            mixup=False,                # Mixup usage
            pad_len=DEFAULT_PAD_LEN,    # Number of frames to pad the start
                                        # and end of videos
            dataset = 'finediving'         # Dataset name
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._split = label_file.split('/')[-1].split('.')[0]
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._dataset = dataset
        self._store_dir = store_dir
        self._store_mode = store_mode
        assert store_mode in ['store', 'load']
        self._clip_len = clip_len
        assert clip_len > 0
        self._stride = stride
        assert stride > 0
        if overlap != 1:
            self._overlap = int((1-overlap) * clip_len)
        else:
            self._overlap = 1
        assert overlap >= 0 and overlap <= 1
        self._dataset_len = dataset_len
        assert dataset_len > 0
        self._pad_len = pad_len
        assert pad_len >= 0

        # Label modifications
        self._radi_displacement = radi_displacement

        #Mixup
        self._mixup = mixup        

        #Frame reader class
        self._frame_reader = FrameReader(frame_dir, modality, dataset = dataset)

        #Store or load clips
        if self._store_mode == 'store':
            self._store_clips()
        elif self._store_mode == 'load':
            self._load_clips()

        self._total_len = len(self._frame_paths)

    def _store_clips(self):
        #Initialize frame paths list
        self._frame_paths = []
        self._labels_store = []
        if self._radi_displacement > 0:
            self._labelsD_store = []
        for video in tqdm(self._labels):
            video_len = int(video['num_frames'])

            labels_file = video['events']

            for base_idx in range(-self._pad_len * self._stride, max(0, video_len - 1 + (2 * self._pad_len - self._clip_len) * self._stride), self._overlap):

                frames_paths = self._frame_reader.load_paths(video['video'], base_idx, base_idx + self._clip_len * self._stride, stride=self._stride)
                
                labels = []
                if self._radi_displacement > 0:
                    labelsD = []
                for event in labels_file:
                    event_frame = event['frame']
                    label_idx = (event_frame - base_idx) // self._stride

                    if self._radi_displacement > 0:
                        if (label_idx >= -self._radi_displacement and label_idx < self._clip_len + self._radi_displacement):
                            label = self._class_dict[event['label']]
                            for i in range(max(0, label_idx - self._radi_displacement), min(self._clip_len, label_idx + self._radi_displacement + 1)):
                                labels.append({'label': label, 'label_idx': i})
                                labelsD.append({'displ': i - label_idx, 'label_idx': i})
                    else:
                        if (label_idx >= -self._dilate_len and label_idx < self._clip_len + self._dilate_len):
                            label = self._class_dict[event['label']]
                            for i in range(max(0, label_idx - self._dilate_len), min(self._clip_len, label_idx + self._dilate_len + 1)):
                                labels.append({'label': label, 'label_idx': i})

                self._frame_paths.append(frames_paths)
                self._labels_store.append(labels)
                if self._radi_displacement > 0:
                    self._labelsD_store.append(labelsD)

        #Save to store
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)

        if not os.path.exists(store_path):
            os.makedirs(store_path)

        with open(store_path + '/frame_paths.pkl', 'wb') as f:
            pickle.dump(self._frame_paths, f)
        with open(store_path + '/labels.pkl', 'wb') as f:
            pickle.dump(self._labels_store, f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'wb') as f:
                pickle.dump(self._labelsD_store, f)
        print('Stored clips to ' + store_path)
        return
    
    def _load_clips(self):
        store_path = os.path.join(self._store_dir, 'LEN' + str(self._clip_len) + 'DIS' + str(self._radi_displacement) + 'SPLIT' + self._split)
        
        with open(store_path + '/frame_paths.pkl', 'rb') as f:
            self._frame_paths = pickle.load(f)
        with open(store_path + '/labels.pkl', 'rb') as f:
            self._labels_store = pickle.load(f)
        if self._radi_displacement > 0:
            with open(store_path + '/labelsD.pkl', 'rb') as f:
                self._labelsD_store = pickle.load(f)
        print('Loaded clips from ' + store_path)
        return

    def _get_one(self):
        #Get random index
        idx = random.randint(0, self._total_len - 1)

        #Get frame_path and labels dict
        frames_path = self._frame_paths[idx]
        dict_label = self._labels_store[idx]
        if self._radi_displacement > 0:
            dict_labelD = self._labelsD_store[idx]

        #Load frames
        frames = self._frame_reader.load_frames(frames_path, pad=True, stride=self._stride)

        #Process labels
        labels = np.zeros(self._clip_len, np.int64)
        for label in dict_label:
            labels[label['label_idx']] = label['label']

        if self._radi_displacement > 0:
            labelsD = np.zeros(self._clip_len, np.int64)
            for label in dict_labelD:
                labelsD[label['label_idx']] = label['displ']

            return {'frame': frames, 'contains_event': int(np.sum(labels) > 0),
                    'label': labels, 'labelD': labelsD}

        return {'frame': frames, 'contains_event': int(np.sum(labels) > 0),
                'label': labels}

    def __getitem__(self, unused):
        ret = self._get_one()
        
        if self._mixup:
            mix = self._get_one()    # Sample another clip
            
            ret['frame2'] = mix['frame']
            ret['contains_event2'] = mix['contains_event']
            ret['label2'] = mix['label']
            if self._radi_displacement > 0:
                ret['labelD2'] = mix['labelD']

        return ret

    def __len__(self):
        return self._dataset_len

    def print_info(self):
        _print_info_helper(self._src_file, self._labels)



class FrameReader:

    def __init__(self, frame_dir, modality, dataset):
        self._frame_dir = frame_dir
        self.modality = modality
        self.dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path) #.float() / 255 -> into model normalization / augmentations
        return img
    
    def load_paths(self, video_name, start, end, stride=1, source_info = None):

        if self.dataset == 'finediving':
            video_name = video_name.replace('__', '/')
            path = os.path.join(self._frame_dir, video_name)
            frame0 = sorted(os.listdir(path))[0]
            ndigits = len(frame0[:-4])
            frame0 = int(frame0[:-4])

        found_start = -1
        pad_start = 0
        pad_end = 0
        for frame_num in range(start, end, stride):

            if frame_num < 0:
                pad_start += 1
                continue

            if pad_end > 0:
                pad_end += 1
                continue
            
            if self.dataset == 'finediving':
                frame = frame0 + frame_num
                frame_path = os.path.join(path, str(frame).zfill(ndigits) + '.jpg')
                base_path = path

            elif (self.dataset == 'fs_comp') | (self.dataset == 'fs_perf'):
                frame = frame_num
                frame_path = os.path.join(self._frame_dir, video_name, 'frame' + str(frame) + '.jpg')
                base_path = os.path.join(self._frame_dir, video_name)
                ndigits = -1
                
            exist_frame = os.path.exists(frame_path)
            if exist_frame & (found_start == -1):
                found_start = frame

            if not exist_frame:
                pad_end += 1

        ret = [base_path, found_start, pad_start, pad_end, ndigits, (end-start) // stride]

        return ret
    
    def load_frames(self, paths, pad=False, stride=1):
        base_path = paths[0]
        start = paths[1]
        pad_start = paths[2]
        pad_end = paths[3]
        ndigits = paths[4]
        length = paths[5]

        ret = []
        if ndigits == -1:
            path = os.path.join(base_path, 'frame')

        else:
            path = base_path + '/'
            _ = [ret.append(self.read_frame(path + str(start + j * stride).zfill(ndigits) + '.jpg')) for j in range(length - pad_start - pad_end)]

        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if pad_start > 0 or (pad and pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, pad_start, pad_end if pad else 0))            

        return ret
    

class ActionSpotVideoDataset(Dataset):

    def __init__(
            self,
            classes,
            label_file,
            frame_dir,
            modality,
            clip_len,
            overlap_len=0,
            stride=1,
            pad_len=DEFAULT_PAD_LEN,
            dataset = 'finediving'
    ):
        self._src_file = label_file
        self._labels = load_json(label_file)
        self._class_dict = classes
        self._video_idxs = {x['video']: i for i, x in enumerate(self._labels)}
        self._clip_len = clip_len
        self._stride = stride
        self._dataset = dataset

        self._frame_reader = FrameReaderVideo(frame_dir, modality, dataset = dataset)

        self._clips = []
        for l in self._labels:
            has_clip = False
            for i in range(
                -pad_len * self._stride,
                max(0, l['num_frames'] - (overlap_len * stride)), \
                # Need to ensure that all clips have at least one frame
                (clip_len - overlap_len) * self._stride
            ):
                has_clip = True
                self._clips.append((l['video'], i))
            assert has_clip, l

    def __len__(self):
        return len(self._clips)

    def __getitem__(self, idx):
        video_name, start = self._clips[idx]

        frames = self._frame_reader.load_frames(
            video_name, start, start + self._clip_len * self._stride, pad=True,
            stride=self._stride)

        return {'video': video_name, 'start': start // self._stride,
                'frame': frames}

    def get_labels(self, video):
        meta = self._labels[self._video_idxs[video]]
        labels_file = meta['events']
        
        num_frames = meta['num_frames']
        num_labels = num_frames // self._stride

        if num_frames % self._stride != 0:
            num_labels += 1
        labels = np.zeros(num_labels, np.int64)
        for event in labels_file:
            frame = event['frame']
            if frame < num_frames:
                labels[frame // self._stride] = self._class_dict[event['label']]
            else:
                print('Warning: {} >= {} is past the end {}'.format(
                    frame, num_frames, meta['video']))
        return labels

    @property
    def videos(self):
        return sorted([
            (v['video'], v['num_frames'] // self._stride,
            v['fps'] / self._stride) for v in self._labels])

    @property
    def labels(self):
        assert self._stride > 0
        if self._stride == 1:
            return self._labels
        else:
            labels = []
            for x in self._labels:
                x_copy = copy.deepcopy(x)
                
                x_copy['fps'] /= self._stride
                x_copy['num_frames'] //= self._stride
                
                for e in x_copy['events']:
                    e['frame'] //= self._stride

                labels.append(x_copy)
            return labels

    def print_info(self):
        num_frames = sum([x['num_frames'] for x in self._labels])
        num_events = sum([len(x['events']) for x in self._labels])
        print('{} : {} videos, {} frames ({} stride), {:0.5f}% non-bg'.format(
            self._src_file, len(self._labels), num_frames, self._stride,
            num_events / num_frames * 100))
        

class FrameReaderVideo:

    def __init__(self, frame_dir, modality, dataset):
        self._frame_dir = frame_dir
        self._modality = modality
        assert self._modality == 'rgb'
        self._dataset = dataset

    def read_frame(self, frame_path):
        img = torchvision.io.read_image(frame_path) #/ 255 -> modified for ActionSpotVideoDataset (to be compatible with train reading without / 255)
        return img

    def load_frames(self, video_name, start, end, pad=False, stride=1, source_info = None):
        ret = []
        n_pad_start = 0
        n_pad_end = 0

        if self._dataset == 'finediving':
            video_name = video_name.replace('__', '/')
            path = os.path.join(self._frame_dir, video_name)
            frame0 = sorted(os.listdir(path))[0]
            ndigits = len(frame0[:-4])
            frame0 = int(frame0[:-4])

        for frame_num in range(start, end, stride):

            if frame_num < 0:
                n_pad_start += 1
                continue
            
            if self._dataset == 'finediving':
                frame_path = os.path.join(path, str(frame0 + frame_num).zfill(ndigits) + '.jpg')

            elif (self._dataset == 'fs_comp') or (self._dataset == 'fs_perf'):
                frame_path = os.path.join(
                    self._frame_dir, video_name, 'frame' + str(frame_num) + '.jpg'
                )
            
            try:
                img = self.read_frame(frame_path)
                ret.append(img)
            except RuntimeError:
                # print('Missing file!', frame_path)
                n_pad_end += 1

        if len(ret) == 0:
            return -1 # Return -1 if no frames were loaded

        # In the multicrop case, the shape is (B, T, C, H, W)
        ret = torch.stack(ret, dim=int(len(ret[0].shape) == 4))

        # Always pad start, but only pad end if requested
        if n_pad_start > 0 or (pad and n_pad_end > 0):
            ret = torch.nn.functional.pad(
                ret, (0, 0, 0, 0, 0, 0, n_pad_start, n_pad_end if pad else 0))
        return ret
    


def _print_info_helper(src_file, labels):
        num_frames = sum([x['num_frames'] for x in labels])
        num_events = sum([len(x['events']) for x in labels])
        print('{} : {} videos, {} frames, {:0.5f}% non-bg'.format(
            src_file, len(labels), num_frames,
            num_events / num_frames * 100))