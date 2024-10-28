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
from SoccerNet.Evaluation.ActionSpotting import average_mAP
import zipfile
from SoccerNet.Evaluation.utils import LoadJsonFromZip
import json
import glob

#Local imports
from util.score import compute_mAPs
from util.io import store_json, store_json_sn, store_json_snb

#Constants
TOLERANCES = [1, 2, 4]
WINDOWS = [1, 3]
TOLERANCES_SN = [3, 6]
WINDOWS_SN = [3, 6]
TOLERANCES_SNB = [6, 12]
WINDOWS_SNB = [6, 12]
WINDOWS_T = [1, 3]
WINDOWS_FG = [1, 3]
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

def process_frame_predictions_challenge(
        dataset, classes, pred_dict, high_recall_score_threshold=0.05
):
    classes_inv = {v: k for k, v in classes.items()}

    fps_dict = {}
    for video, _, fps in dataset.videos:
        fps_dict[video] = fps

    pred_events = []
    pred_events_high_recall = []
    pred_scores = {}
    h = 0
    for video, (scores, support) in (sorted(pred_dict.items())):
        #h += 1
        #if h > 50:
        #    break
        if np.min(support) == 0:
            support[support == 0] = 1
        assert np.min(support) > 0, (video, support.tolist())
        scores /= support[:, None]
        pred = np.argmax(scores, axis=1)

        pred_scores[video] = scores.tolist()

        events = []
        events_high_recall = []
        for i in range(pred.shape[0]):

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

    return pred_events, pred_events_high_recall, pred_scores

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

    if dataset._dataset == 'soccernet':
        tolerances = TOLERANCES_SN
        windows = WINDOWS_SN

    if dataset._dataset == 'soccernetball':
        tolerances = TOLERANCES_SNB
        windows = WINDOWS_SNB

    if dataset._dataset == 'tennis':
        windows = WINDOWS_T

    if dataset._dataset == 'finegym':
        windows = WINDOWS_FG

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

    if split != 'CHALLENGE':
        err, f1, pred_events, pred_events_high_recall, pred_scores = \
            process_frame_predictions(dataset, classes, pred_dict, high_recall_score_threshold=0.01)
    else:
        pred_events, pred_events_high_recall, pred_scores = \
            process_frame_predictions_challenge(dataset, classes, pred_dict, high_recall_score_threshold=0.01)

    if not test:
        pred_events_high_recall = non_maximum_supression(pred_events_high_recall, window = windows[0], threshold = 0.10)
        mAPs, _ = compute_mAPs(dataset.labels, pred_events_high_recall, tolerances=tolerances, printed = True)
        avg_mAP = np.mean(mAPs)
        return avg_mAP
    
    else:

        if split != 'CHALLENGE':

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
            mAPs, tolerances = compute_mAPs(dataset.labels, pred_events_high_recall_nms, tolerances=tolerances, printed = printed)
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
                
                if dataset._dataset == 'soccernet':
                    store_json_sn(save_pred, pred_events_high_recall_store, stride = dataset._stride)
                if dataset._dataset == 'soccernetball':
                    store_json_snb(save_pred, pred_events_high_recall_store, stride = dataset._stride)

            return mAPs, tolerances
        
        else:
            pred_events_high_recall_store = soft_non_maximum_supression(pred_events_high_recall, window = windows[1], threshold=0.01)
            print('Storing predictions Challenge with SNMS')
            store_json_snb(save_pred, pred_events_high_recall, stride = dataset._stride)
            return None, None


def valMAP_SN(labels, preds, framerate = 25, metric = "tight", version = 2):

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    for i in range(len(labels)):
        label = labels[i].numpy()[:, 1:]
        pred = preds[i].numpy()[:, 1:]

        targets_numpy.append(label)
        detections_numpy.append(pred)

        closest_numpy = np.zeros(label.shape) - 1
        # Get the closest action index
        for c in np.arange(label.shape[-1]):
            indexes = np.where(label[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = label[indexes[i], c]
        closests_numpy.append(closest_numpy)

    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])  # np.arange(1)*1 + 1
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])

    # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible if version == 2 else None,
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version == 2 else None,
        "a_mAP_unshown": a_mAP_unshown if version == 2 else None,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version == 2 else None,
    }
    return results

def evaluate_SNB(label_path, pred_path, split = 'test'):
    games = {
        'train': ["england_efl/2019-2020/2019-10-01 - Leeds United - West Bromwich",
            "england_efl/2019-2020/2019-10-01 - Hull City - Sheffield Wednesday",
            "england_efl/2019-2020/2019-10-01 - Brentford - Bristol City",
            "england_efl/2019-2020/2019-10-01 - Blackburn Rovers - Nottingham Forest"],
        'val' : ["england_efl/2019-2020/2019-10-01 - Middlesbrough - Preston North End"],
        'test': ["england_efl/2019-2020/2019-10-01 - Stoke City - Huddersfield Town",
            "england_efl/2019-2020/2019-10-01 - Reading - Fulham"],
        'challenge': ["england_efl/2019-2020/2019-10-02 - Cardiff City - Queens Park Rangers",
            "england_efl/2019-2020/2019-10-01 - Wigan Athletic - Birmingham City"]
        }

    return aux_evaluate(label_path, pred_path, list_games = games[split], prediction_file = 'results_spotting.json',
            version = 2, metric = 'at1', num_classes = 12, label_files = 'Labels-ball.json', 
            dataset = 'Ball', framerate=25)

def aux_evaluate(SoccerNet_path, Predictions_path, list_games, prediction_file="results_spotting.json", version=2,
            framerate=2, metric="loose", label_files="Labels-v2.json", num_classes=17, dataset="SoccerNet"):

    targets_numpy = list()
    detections_numpy = list()
    closests_numpy = list()

    EVENT_DICTIONARY = {"PASS":0, "DRIVE":1, "HEADER":2, "HIGH PASS":3, "OUT":4, "CROSS":5, "THROW IN":6, "SHOT":7, "BALL PLAYER BLOCK":8, 
                        "PLAYER SUCCESSFUL TACKLE":9, "FREE KICK":10, "GOAL":11}
        

    for game in tqdm(list_games):

        if zipfile.is_zipfile(SoccerNet_path):
            labels = LoadJsonFromZip(SoccerNet_path, os.path.join(game, label_files))
        else:
            labels = json.load(open(os.path.join(SoccerNet_path, game, label_files)))
        # convert labels to vector
        label_half_1 = label2vector(
            labels, num_classes=num_classes, version=version, EVENT_DICTIONARY=EVENT_DICTIONARY, framerate=framerate)
        # print(version)

        # infer name of the prediction_file
        if prediction_file is None:
            if zipfile.is_zipfile(Predictions_path):
                with zipfile.ZipFile(Predictions_path, "r") as z:
                    for filename in z.namelist():
                        #       print(filename)
                        if filename.endswith(".json"):
                            prediction_file = os.path.basename(filename)
                            break
            else:
                for filename in glob.glob(os.path.join(Predictions_path, "*/*/*/*.json")):
                    prediction_file = os.path.basename(filename)
                    # print(prediction_file)
                    break

        # Load predictions
        if zipfile.is_zipfile(Predictions_path):
            predictions = LoadJsonFromZip(Predictions_path, os.path.join(game, prediction_file))
        else:
            predictions = json.load(open(os.path.join(Predictions_path, game, prediction_file)))
        # convert predictions to vector
        predictions_half_1 = predictions2vector(
            predictions, num_classes=num_classes, version=version, EVENT_DICTIONARY=EVENT_DICTIONARY,
            framerate=framerate)


        targets_numpy.append(label_half_1)
        detections_numpy.append(predictions_half_1)

        closest_numpy = np.zeros(label_half_1.shape) - 1
        # Get the closest action index
        for c in np.arange(label_half_1.shape[-1]):
            indexes = np.where(label_half_1[:, c] != 0)[0].tolist()
            if len(indexes) == 0:
                continue
            indexes.insert(0, -indexes[0])
            indexes.append(2 * closest_numpy.shape[0])
            for i in np.arange(len(indexes) - 2) + 1:
                start = max(0, (indexes[i - 1] + indexes[i]) // 2)
                stop = min(closest_numpy.shape[0], (indexes[i] + indexes[i + 1]) // 2)
                closest_numpy[start:stop, c] = label_half_1[indexes[i], c]
        closests_numpy.append(closest_numpy)


    if metric == "loose":
        deltas = np.arange(12) * 5 + 5
    elif metric == "tight":
        deltas = np.arange(5) * 1 + 1
    elif metric == "at1":
        deltas = np.array([1])  # np.arange(1)*1 + 1
    elif metric == "at2":
        deltas = np.array([2])
    elif metric == "at3":
        deltas = np.array([3])
    elif metric == "at4":
        deltas = np.array([4])
    elif metric == "at5":
        deltas = np.array([5])
        # Compute the performances
    a_mAP, a_mAP_per_class, a_mAP_visible, a_mAP_per_class_visible, a_mAP_unshown, a_mAP_per_class_unshown = (
        average_mAP(targets_numpy, detections_numpy, closests_numpy, framerate, deltas=deltas)
    )

    results = {
        "a_mAP": a_mAP,
        "a_mAP_per_class": a_mAP_per_class,
        "a_mAP_visible": a_mAP_visible if version == 2 else None,
        "a_mAP_per_class_visible": a_mAP_per_class_visible if version == 2 else None,
        "a_mAP_unshown": a_mAP_unshown if version == 2 else None,
        "a_mAP_per_class_unshown": a_mAP_per_class_unshown if version == 2 else None,
    }
    return results

def label2vector(labels, num_classes=17, framerate=2, version=2, EVENT_DICTIONARY={}):

    vector_size = 120*60*framerate

    label_half1 = np.zeros((vector_size, num_classes))

    for annotation in labels["annotations"]:

        time = annotation["gameTime"]
        event = annotation["label"]

        half = int(time[0])

        minutes = int(time[-5:-3])
        seconds = int(time[-2::])
        # annotation at millisecond precision
        if "position" in annotation:
            frame = int(framerate * ( int(annotation["position"])/1000 ))
        # annotation at second precision
        else:
            frame = framerate * ( seconds + 60 * minutes )


        if version == 2:
            if event not in EVENT_DICTIONARY:
                continue
            label = EVENT_DICTIONARY[event]
        elif version == 1:
            # print(event)
            # label = EVENT_DICTIONARY[event]
            if "card" in event: label = 0
            elif "subs" in event: label = 1
            elif "soccer" in event: label = 2
            else:
                # print(event)
                continue
        # print(event, label, half)

        value = 1
        if "visibility" in annotation.keys():
            if annotation["visibility"] == "not shown":
                value = -1

        if half == 1:
            frame = min(frame, vector_size-1)
            label_half1[frame][label] = value

    return label_half1

def predictions2vector(predictions, num_classes=17, version=2, framerate=2, EVENT_DICTIONARY={}):


    vector_size = 120*60*framerate

    prediction_half1 = np.zeros((vector_size, num_classes))-1

    for annotation in predictions["predictions"]:

        time = int(annotation["position"])
        event = annotation["label"]

        half = int(annotation["half"])

        frame = int(framerate * ( time/1000 ))

        if version == 2:
            if event not in EVENT_DICTIONARY:
                continue
            label = EVENT_DICTIONARY[event]
        elif version == 1:
            label = EVENT_DICTIONARY[event]
            # print(label)
            # EVENT_DICTIONARY_V1[l]
            # if "card" in event: label=0
            # elif "subs" in event: label=1
            # elif "soccer" in event: label=2
            # else: continue

        value = annotation["confidence"]

        if half == 1:
            frame = min(frame, vector_size-1)
            prediction_half1[frame][label] = value

    return prediction_half1