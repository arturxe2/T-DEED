import os
import re
import json
import pickle
import gzip

FPS_SN = 25

def load_json(fpath):
    with open(fpath) as fp:
        return json.load(fp)


def store_json(fpath, obj, pretty=False):
    kwargs = {}
    if pretty:
        kwargs['indent'] = 2
        kwargs['sort_keys'] = True
    with open(fpath, 'w') as fp:
        json.dump(obj, fp, **kwargs)

def store_json_sn(pred_path, pred, stride = 1):

    i = 0
    for game in pred:
        if i % 2 == 0:
            gameDict = dict()
            gameDict['UrlLocal'] = game['video']
            gameDict['predictions'] = []
        for event in game['events']:
            eventDict = dict()
            position = int(event['frame'] / FPS_SN * 1000 * stride)
            eventDict['gameTime'] = '{} - {}:{}'.format((i % 2) + 1, position // 60000, int((position % 60000) // 1000))
            eventDict['label'] = event['label']
            eventDict['position'] = position
            eventDict['confidence'] = event['score']
            eventDict['half'] = (i % 2) + 1
            gameDict['predictions'].append(eventDict)

        if (i % 2) == 1:
            path = os.path.join('/'.join(pred_path.split('/')[:-1]) + '/preds', '/'.join(game['video'].split('/')[:-1]))
            if not os.path.exists(path):
                os.makedirs(path)
            with open(path + '/results_spotting.json', 'w') as fp:
                json.dump(gameDict, fp, indent=4)
        
        i += 1

def store_json_snb(pred_path, pred, stride = 1):
    for game in pred:
        gameDict = dict()
        gameDict['UrlLocal'] = game['video']
        gameDict['predictions'] = []
        for event in game['events']:
            eventDict = dict()
            position = int(event['frame'] / FPS_SN * 1000 * stride)
            eventDict['gameTime'] = '1 - {}:{}'.format(position // 60000, int((position % 60000) // 1000))
            eventDict['label'] = event['label']
            eventDict['position'] = position
            eventDict['confidence'] = event['score']
            eventDict['half'] = 1
            gameDict['predictions'].append(eventDict)

        path = os.path.join('/'.join(pred_path.split('/')[:-1]) + '/preds', game['video'])
        if not os.path.exists(path):
            os.makedirs(path)
        with open(path + '/results_spotting.json', 'w') as fp:
            json.dump(gameDict, fp, indent=4)

def load_text(fpath):
    lines = []
    with open(fpath, 'r') as fp:
        for l in fp:
            l = l.strip()
            if l:
                lines.append(l)
    return lines