# Setting up SoccerNet Ball

This directory contains the splits converted from the original SoccerNet Ball Action Spotting dataset, available at: https://www.soccer-net.org/tasks/ball-action-spotting.

To download the videos follow instructions provided in [SoccerNet](https://www.soccer-net.org/tasks/ball-action-spotting), and to generate the folder structure for frames, use the provided script [extract_frames_snb.py] adapted from E2E-Spot.

To train T-DEED for the challenge split, modify `train.json` and `val.json` for `train_challenge.json` and `val_challenge`. 

Frames are extracted at a resolution of 796x448, and frame naming convention is as follows:

```
data-folder
└───england_efl
    └───2019-2020
        └───2019-10-01 - Blackburn Rovers - Nottingham Forest
        |frame0.jpg
        |frame1.jpg
        |...
        └───2019-10-01 - Brentford - Bristol City
        |frame0.jpg
        |frame1.jpg
        |...
```

---
