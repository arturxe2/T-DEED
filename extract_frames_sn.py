import os
import argparse
import cv2
import moviepy.editor
from tqdm import tqdm
from multiprocessing import Pool
cv2.setNumThreads(0)
from SoccerNet.Downloader import getListGames

'''
This script extracts frames from SoccerNetv2 Action Spotting dataset by introducing the path where the downloaded videos are (at 720 resolution), the path to
write the frames, the sample fps, and the number of workers to use. The script will create a folder for each video in the out_dir path and save the frames as .jpg files in
the desired resolution.

Example usage:

python extract_frames_sn.py --video_dir video_dir
        --out_dir out_dir
        --sample_fps 12.5 --num_workers 4

Extracted frames with a resolution of 796x448 with a sample fps of 12.5 take approximately 7.5G per game.
Extracted frames with a resolution of 398x224 with a sample fps of 12.5 take approximately 1.1T.
'''


RECALC_FPS_ONLY = False
FRAME_RETRY_THRESHOLD = 1000

### For SoccerNet training (not SoccerNet Ball joint training) download frames at 398x224
#TARGET_HEIGHT = 224
#TARGET_WIDTH = 398
TARGET_HEIGHT = 448
TARGET_WIDTH = 796

SPLIT = ['train', 'valid', 'test']


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video_dir', help='Path to the downloaded videos')
    parser.add_argument('-o', '--out_dir',
                        help='Path to write frames. Dry run if None.')
    parser.add_argument('--sample_fps', type=float, default=2)
    parser.add_argument('--recalc_fps', action='store_true')
    parser.add_argument('-j', '--num_workers', type=int,
                        default=os.cpu_count() // 4)
    return parser.parse_args()


def get_duration(video_path):
    # Copied from SoccerNet repo
    return moviepy.editor.VideoFileClip(video_path).duration


def worker(args):
    video_name, video_path, out_dir, sample_fps = args

    def get_stride(src_fps):
        if sample_fps <= 0:
            stride = 1
        else:
            stride = int(src_fps / sample_fps)
        return stride
    
    vc = cv2.VideoCapture(video_path)
    fps = vc.get(cv2.CAP_PROP_FPS)
    if fps != 25:
        print('FPS is not 25:', video_name)
        fps = 25
    num_frames = int(vc.get(cv2.CAP_PROP_FRAME_COUNT))
    w = int(vc.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(vc.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    oh = TARGET_HEIGHT
    ow = TARGET_WIDTH

    time_in_s = get_duration(video_path)

    fps_path = None

    if out_dir is not None:
        fps_path = os.path.join(out_dir, 'fps.txt')
        os.makedirs(out_dir, exist_ok=True)

    #check real number of frames: (there are some inconsistencies in some games)
    nframes = 0
    while True:
        ret, frame = vc.read()
        if not ret:
            break
        nframes += 1

    vc.release()

    vc = cv2.VideoCapture(video_path)

    if num_frames - nframes > FRAME_RETRY_THRESHOLD:
        effective_fps = fps * nframes / num_frames
        print('Not aligned frames, modified effective fps:', effective_fps)
    else:
        effective_fps = fps

    # if the effective fps is the same as the fps, we can just read the frames
    if effective_fps == fps: 
    
        not_done = True
        stride = get_stride(fps)
        while not_done:
            est_out_fps = fps / stride
            print('{} -- effective fps: {} (stride: {})'.format(
                video_name, est_out_fps, stride))

            out_frame_num = 0
            i = 0
            while True:
                ret, frame = vc.read()
                if not ret:
                    # fps and num_frames are wrong
                    print('problem reading or finished')
                    if i + FRAME_RETRY_THRESHOLD < num_frames:
                        print('Problem in the video.')

                    else:
                        print('Finished or semi: {} -- {} / {}'.format(
                            video_path, i, num_frames))
                    
                    not_done = False

                    break

                if i % stride == 0:
                    #if out_frame_num >= min_frame and out_frame_num <= max_frame: # -> in case we only want to store frames around actions
                    if True:
                        if not RECALC_FPS_ONLY:
                            if frame.shape[0] != oh or frame.shape[1] != ow:
                                frame = cv2.resize(frame, (ow, oh))

                            if out_dir is not None:
                                frame_path = os.path.join(
                                    out_dir, 'frame{}.jpg'.format(out_frame_num))
                                cv2.imwrite(frame_path, frame)

                out_frame_num += 1
                i += 1

    # In case number of frames is not totally aligned with the fps
    else:
        print('Video with strange framerate')
        not_done = True
        stride = get_stride(fps)
        while not_done:
            print('{} -- effective fps: {} (stride: {})'.format(
                    video_name, effective_fps, stride))

            out_frame_num = 0
            i = 0
            while True:
                ret, frame = vc.read()
                if not ret:
                    # fps and num_frames are wrong
                    print('finished with total numer of frames: ' + str(out_frame_num))
                    not_done = False

                    break
                    
                aux_i = i * fps / effective_fps
                if aux_i > out_frame_num:
                    if True:
                        if not RECALC_FPS_ONLY:
                            if frame.shape[0] != oh or frame.shape[1] != ow:
                                frame = cv2.resize(frame, (ow, oh))
                                
                            if out_dir is not None:
                                frame_path = os.path.join(
                                    out_dir, 'frame{}.jpg'.format(out_frame_num))
                                cv2.imwrite(frame_path, frame)
                    out_frame_num += stride
                i += 1

    vc.release()

    out_fps = fps / get_stride(fps)
    if fps_path is not None:
        with open(fps_path, 'w') as fp:
            fp.write(str(out_fps))
    print('{} - done'.format(video_name))


def main(args, games = None):
    
    video_dir = args.video_dir
    out_dir = args.out_dir
    sample_fps = args.sample_fps
    recalc_fps = args.recalc_fps
    num_workers = args.num_workers

    global RECALC_FPS_ONLY
    RECALC_FPS_ONLY = recalc_fps

    worker_args = []
    for game in games:
        for video_file in os.listdir(os.path.join(video_dir, game)):
            
            if (video_file.endswith('.mkv') | video_file.endswith('.mp4')):
                half = os.path.splitext(video_file)[0].replace(
                    '_720p', '')
                worker_args.append((os.path.join(game, video_file), 
                                    os.path.join(video_dir, game, video_file),
                                    os.path.join(out_dir, game, 'half' + str(half)),
                                    sample_fps))

    with Pool(num_workers) as p:
        for _ in tqdm(p.imap_unordered(worker, worker_args),
                    total = len(worker_args)):
            pass

    print('Done!')



if __name__ == '__main__':
    args = get_args()
    args.out_dir = args.out_dir + str(TARGET_HEIGHT)
    games = getListGames(SPLIT)

    main(args, games = games)