import os

from util.io import load_text


DATASETS = [
    'tennis',
    'fs_perf',
    'fs_comp',
    'finediving',
    'finegym',
    'soccernetv2',
    'soccernetball'
]


def load_classes(file_name):
    return {x: i + 1 for i, x in enumerate(load_text(file_name))}

def read_fps(video_frame_dir):
    with open(os.path.join(video_frame_dir, 'fps.txt')) as fp:
        return float(fp.read())