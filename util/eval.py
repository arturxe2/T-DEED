"""
File containing main evaluation functions
"""

#Standard imports
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from collections import defaultdict
import copy
from tabulate import tabulate
import os

#Local imports
from util.score import compute_mAPs
from util.io import store_json

#Constants
TOLERANCES = [1, 2, 4]
WINDOWS = [1, 3]
INFERENCE_BATCH_SIZE = 4

class ErrorStat:

    def __init__(self):
        self._total = 0
        self._err = 0

    def update(self, true, pred):
        self._err += np.sum(true != pred)
        self._total += true.shape[0]

    def get(self):
        return self._err / self._total

    def get_acc(self):
        return 1. - self._get()
    
class ForegroundF1:

    def __init__(self):
        self._tp = defaultdict(int)
        self._fp = defaultdict(int)
        self._fn = defaultdict(int)

    def update(self, true, pred):
        if pred != 0:
            if true != 0:
                self._tp[None] += 1
            else:
                self._fp[None] += 1

            if pred == true:
                self._tp[pred] += 1
            else:
                self._fp[pred] += 1
                if true != 0:
                    self._fn[true] += 1
        elif true != 0:
            self._fn[None] += 1
            self._fn[true] += 1

    def get(self, k):
        return self._f1(k)

    def tp_fp_fn(self, k):
        return self._tp[k], self._fp[k], self._fn[k]

    def _f1(self, k):
        denom = self._tp[k] + 0.5 * self._fp[k] + 0.5 * self._fn[k]
        if denom == 0:
            assert self._tp[k] == 0
            denom = 1
        return self._tp[k] / denom

def process_frame_predictions(dataset, classes, pred_dict, high_recall_score_threshold=0.01):
    
    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}
    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    err = ErrorStat()
    f1 = ForegroundF1()

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}
    h = 0
    for video, (scores, support) in (sorted(pred_dict.items())):
        label = dataset.get_labels(video)
        if np.min(support) == 0:
            support[support == 0] = 1
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        pred = np.argmax(scores, axis=1)
        err.update(label, pred)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        for i in range(pred.shape[0]):
            f1.update(label[i], pred[i])

            if pred[i] != 0:
                events.append({
                    'label': classes_inv[pred[i]],
                    'frame': i,
                    'score': scores[i, pred[i]].item()
                })

            for j in classes_inv:
                if scores[i, j] >= high_recall_score_threshold:
                    events_high_recall.append({
                        'label': classes_inv[j],
                        'frame': i,
                        'score': scores[i, j].item()
                    })

        pred_events.append({
            'video': video, 'events': events,
            'fps': fps_dict[video]})
        pred_events_high_recall.append({
            'video': video, 'events': events_high_recall,
            'fps': fps_dict[video]})
        
    return err, f1, pred_events, pred_events_high_recall, pred_scores

def non_maximum_supression(pred, window, threshold = 0.0):
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            if type(window) is not list:
                class_window = window
            else:
                class_window = window[i]
                i += 1
            while(len(v) > 0):
                e1 = max(v, key=lambda x:x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                v.pop(pos1)
                list_pos = [pos for pos, e in enumerate(v) if ((e['frame'] >= e1['frame']-class_window) & (e['frame'] <= e1['frame']+class_window))]
                for pos in list_pos[::-1]: #reverse order to avoid movement of positions in the list
                    v.pop(pos)

        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred

def soft_non_maximum_supression(pred, window, threshold = 0.01):
    preds = copy.deepcopy(pred)
    new_pred = []
    for video_pred in preds:
        events_by_label = defaultdict(list)
        for e in video_pred['events']:
            events_by_label[e['label']].append(e)

        events = []
        i = 0
        for v in events_by_label.values():
            if type(window) is not list:
                class_window = window
            else:
                class_window = window[i]
                i += 1
            while(len(v) > 0):
                e1 = max(v, key=lambda x:x['score'])
                if e1['score'] < threshold:
                    break
                pos1 = [pos for pos, e in enumerate(v) if e['frame'] == e1['frame']][0]
                events.append(copy.deepcopy(e1))
                list_pos = [pos for pos, e in enumerate(v) if ((e['frame'] >= e1['frame']-class_window) & (e['frame'] <= e1['frame']+class_window))]
                for pos in list_pos:
                    v[pos]['score'] = v[pos]['score'] * (np.abs(e1['frame'] - v[pos]['frame'])) ** 2 / ((class_window+0) ** 2)
                v.pop(pos1)

        events.sort(key=lambda x: x['frame'])
        new_video_pred = copy.deepcopy(video_pred)
        new_video_pred['events'] = events
        new_video_pred['num_events'] = len(events)
        new_pred.append(new_video_pred)
    return new_pred


def evaluate(model, dataset, split, classes, save_pred=None, printed = True, 
            test = False, augment=False):
    
    tolerances = TOLERANCES
    windows = WINDOWS

    pred_dict = {}
    for video, video_len, _ in dataset.videos:
        pred_dict[video] = (
            np.zeros((video_len, len(classes) + 1), np.float32),
            np.zeros(video_len, np.int32))

    # Do not up the batch size if the dataset augments
    batch_size = 1 if augment else INFERENCE_BATCH_SIZE
    
    h = 0
    for clip in tqdm(DataLoader(
            dataset, num_workers=4 * 2, pin_memory=True,
            batch_size=batch_size
    )):
            
        if batch_size > 1:
            # Batched by dataloader
            _, batch_pred_scores = model.predict(clip['frame'])

            for i in range(clip['frame'].shape[0]):
                video = clip['video'][i]
                scores, support = pred_dict[video]
                pred_scores = batch_pred_scores[i]
                start = clip['start'][i].item()
                if start < 0:
                    pred_scores = pred_scores[-start:, :]
                    start = 0
                end = start + pred_scores.shape[0]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:end - start, :]

                scores[start:end, :] += pred_scores
                support[start:end] += (pred_scores.sum(axis=1) != 0) * 1

        else:
            # Batched by dataset
            scores, support = pred_dict[clip['video'][0]]

            start = clip['start'][0].item()
            _, pred_scores = model.predict(clip['frame'])
            if start < 0:
                pred_scores = pred_scores[:, -start:, :]
                start = 0
            end = start + pred_scores.shape[1]
            if end >= scores.shape[0]:
                end = scores.shape[0]
                pred_scores = pred_scores[:,:end - start, :]

            scores[start:end, :] += np.sum(pred_scores, axis=0)
            support[start:end] += pred_scores.shape[0]

            #Additional view with horizontal flip
            for i in range(1):
                start = clip['start'][0].item()
                _, pred_scores = model.predict(clip['frame'], augment_inference = True)
                if start < 0:
                    pred_scores = pred_scores[:, -start:, :]
                    start = 0
                end = start + pred_scores.shape[1]
                if end >= scores.shape[0]:
                    end = scores.shape[0]
                    pred_scores = pred_scores[:,:end - start, :]

                scores[start:end, :] += np.sum(pred_scores, axis=0)
                support[start:end] += pred_scores.shape[0]

    err, f1, pred_events, pred_events_high_recall, pred_scores = \
        process_frame_predictions(dataset, classes, pred_dict, high_recall_score_threshold=0.01)

    if not test:
        pred_events_high_recall = non_maximum_supression(pred_events_high_recall, window = windows[0], threshold = 0.05)
        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall, tolerances=tolerances, printed = True)
        avg_mAP = np.mean(mAPs)
        return avg_mAP
    
    else:

        print('=== Results on {} (w/o NMS) ==='.format(split))
        print('Error (frame-level): {:0.2f}\n'.format(err.get() * 100))

        def get_f1_tab_row(str_k):
            k = classes[str_k] if str_k != 'any' else None
            return [str_k, f1.get(k) * 100, *f1.tp_fp_fn(k)]
        rows = [get_f1_tab_row('any')]
        for c in sorted(classes):
            rows.append(get_f1_tab_row(c))

        print(tabulate(rows, headers=['Exact frame', 'F1', 'TP', 'FP', 'FN'],
                        floatfmt='0.2f'))
        print()

        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall, tolerances=tolerances, printed = printed)
        avg_mAP = np.mean(mAPs)

        print('=== Results on {} (w/ NMS{}) ==='.format(split, str(windows[0])))
        pred_events_high_recall_nms = non_maximum_supression(pred_events_high_recall, window = windows[0], threshold=0.01)
        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall_nms, tolerances=tolerances, printed = printed)
        avg_mAP_nms = np.mean(mAPs)

        print('=== Results on {} (w/ SNMS{}) ==='.format(split, str(windows[1])))
        pred_events_high_recall_snms = soft_non_maximum_supression(pred_events_high_recall, window = windows[1], threshold=0.01)
        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall_snms, tolerances=tolerances, printed = printed)
        avg_mAP_snms = np.mean(mAPs)


        if avg_mAP_snms > avg_mAP_nms:
            print('Storing predictions with SNMS')
            pred_events_high_recall_store = pred_events_high_recall_snms
        else:
            print('Storing predictions with NMS')
            pred_events_high_recall_store = pred_events_high_recall_nms
        
        if save_pred is not None:
            if not os.path.exists('/'.join(save_pred.split('/')[:-1])):
                os.makedirs('/'.join(save_pred.split('/')[:-1]))
            store_json(save_pred + '.json', pred_events_high_recall_store)

        return avg_mAP_snms