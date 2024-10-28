# Setting up SoccerNet Ball

This directory contains the splits converted from the original SoccerNet Action Spotting dataset, available at: https://www.soccer-net.org/tasks/action-spotting.

To download the videos follow instructions provided in [SoccerNet](https://www.soccer-net.org/tasks/action-spotting), and to generate the folder structure for frames, use the provided script [extract_frames_sn.py](../../extract_frames_snb.py) adapted from E2E-Spot.

Frames are extracted at a resolution of 796x448, and frame naming convention is as follows:

```
data-folder
└───england_efl
    └───2014-2015
        └───2015-02-21 - 18-00 Chelsea 1 - 1 Burnley
            └───half1
                |frame0.jpg
                |frame1.jpg
                |...
            └───half2
                |frame0.jpg
                |frame1.jpg
                |...
        └───2015-02-21 - 18-00 Crystal Palace 1 - 2 Arsenal
            └───half1
                |frame0.jpg
                |frame1.jpg
                |...
            └───half2
                |frame0.jpg
                |frame1.jpg
                |...
```

---