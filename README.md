# T-DEED
### T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event Spotting in Sports Videos (CVsports '24)
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
[![arXiv](https://img.shields.io/badge/arXiv-2404.05392-red)](https://arxiv.org/abs/2404.05392)
[![ProjectPage](https://img.shields.io/badge/Project%20Page-green)](https://arturxe2.github.io/projects/T-DEED/)

This repository contains the PyTorch implementation of the paper:

**T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event Spotting in Sports Videos**<br>
*Artur Xarles, Sergio Escalera, Thomas B. Moeslund, and Albert Clapés*<br>
**10th International Workshop on Computer Vision in Sports (CVsports) at CVPR 2024**<br>


## Overview

This repository builds upon the original [T-DEED implementation](https://github.com/arturxe2/T-DEED) to evaluate the model across additional PES datasets and the broader Action Spotting task, including SoccerNet Action Spotting (SNAS) and SoccerNet Ball Action Spotting (SNBAS). Notably, this implementation was also used to achieve the 1st place in the 2024 SNBAS Challenge.

It contains two distinct releases: the first corresponds to the initial release for the paper, and the second includes an extended evaluation on additional PES datasets along with adaptations for the broader Action Spotting task in SoccerNet.


## Environment

You can install the required packages for the project using the following command, with `requirements.txt` specifying the versions of the various packages:

```
pip install -r requirements.txt
```

## Data

Refer to the README files in the [data](/data/) directory for pre-processing and setup instructions. 


## Execution

The `train_tdeed.py` file is designed to train and evaluate T-DEED based on the settings specified in the chosen configuration file. You can execute the file using the following command:

```
python3 train_tdeed.py --model <model_name>
```

Here, `<model_name>` follows the format `<dataset>_<name>`, where `<dataset>` is one of the possible datasets (FigureSkatingComp, FigureSkatingPerf, FineDiving, FineGym, Tennis, SoccerNet, or SoccerNetBall), and `<name>` can be chosen freely but must match the name specified in the configuration file located in the config directory.

For example, to use the FineDiving dataset with the small model (200MF), you would run:

```
python3 train_tdeed.py --model FineDiving_small
```

You can control whether to train the whole model or just evaluate it using the `only_test` parameter in the configuration file. For additional details on configuration options, refer to the README in the [config](/config/) directory.


Before running the model, ensure that you have downloaded the dataset frames and updated the directory-related configuration parameters in the relevant [config](/config/) files, as well as in the [SN](/data/soccernet/labels_path.txt) and [SNB](/data/soccernetball/labels_path.txt) label files. Additionally, make sure to run the script once with the `mode` parameter set to `store` to generate and save the clip partitions. After this initial run, you can set the `mode` to `load` to reuse the saved partitions for subsequent executions.

## Trained models

Model checkpoints can be found at this [link](https://drive.google.com/drive/folders/1sxZalU_hCwL8ITZCU9VqSWE8dB94lJty?usp=drive_link), and configurations are available in the [config](/config/) directory. To use the checkpoints, place the checkpoint file in the [checkpoints](/checkpoints/) directory, maintaining the same structure as shown with `FineDiving_small`.

*There has been a slight improvement in the results of some datasets compared to those reported in the paper.\
**Note: Some results may slightly vary from those reported in the paper, as the paper's results are the average of two executions with different seeds, while here we only include one of the model's checkpoints.

## Contact

If you have any questions related to the code, feel free to contact arturxe@gmail.com.

## References

If you find our work useful, please consider citing our paper.
```
@inproceedings{xarles2024t,
  title={T-DEED: Temporal-Discriminability Enhancer Encoder-Decoder for Precise Event Spotting in Sports Videos},
  author={Xarles, Artur and Escalera, Sergio and Moeslund, Thomas B and Clap{\'e}s, Albert},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3410--3419},
  year={2024}
}
```
