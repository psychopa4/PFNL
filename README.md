# Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations
This work is based on [Wang et al](https://github.com/psychopa4/MMCNN) and [Tao et al](https://github.com/jiangsutx/SPMC_VideoSR).

This work has tried to rebuild various state-of-the-art video SR methods, including [VESPCN](http://openaccess.thecvf.com/content_cvpr_2017/html/Caballero_Real-Time_Video_Super-Resolution_CVPR_2017_paper.html), 
[RVSR-LTD](http://openaccess.thecvf.com/content_iccv_2017/html/Liu_Robust_Video_Super-Resolution_ICCV_2017_paper.html), 
[MCResNet](https://www.researchgate.net/publication/313830988_Video_Super-Resolution_via_Motion_Compensation_and_Deep_Residual_Learning), 
[DRVSR](http://openaccess.thecvf.com/content_iccv_2017/html/Tao_Detail-Revealing_Deep_Video_ICCV_2017_paper.html), 
[FRVSR](http://openaccess.thecvf.com/content_cvpr_2018/html/Sajjadi_Frame-Recurrent_Video_Super-Resolution_CVPR_2018_paper.html), 
[DUFVSR](http://openaccess.thecvf.com/content_cvpr_2018/html/Jo_Deep_Video_Super-Resolution_CVPR_2018_paper.html) and 
[PFNL](http://openaccess.thecvf.com/content_ICCV_2019/html/Yi_Progressive_Fusion_Video_Super-Resolution_Network_via_Exploiting_Non-Local_Spatio-Temporal_Correlations_ICCV_2019_paper.html).

## Datasets
We have selected [MM522 dataset](https://github.com/psychopa4/MMCNN) for training and collected another 20 sequences for evaluation, and in consider of copyright, the datasets should only be used for study.

The datasets can be downloaded from Google Drive, [train](https://drive.google.com/open?id=1xPMYiA0JwtUe9GKiUa4m31XvDPnX7Juu) and [evaluation](https://drive.google.com/file/d/1Px0xAE2EUzXbgfDJZVR2KfG7zAk7wPZO/view?usp=sharing).

Note that the [training](https://drive.google.com/open?id=1xPMYiA0JwtUe9GKiUa4m31XvDPnX7Juu) dataset provides Ground Truth images and Bicubic downsampling LR images, while the [evaluation](https://drive.google.com/file/d/1Px0xAE2EUzXbgfDJZVR2KfG7zAk7wPZO/view?usp=sharing) dataset provides Gaussian blur and downsampling images. Thus, please refer to ./model/base_model.py for generating Gaussian blur and downsampling images from Ground Truth images.

Unzip the training dataset to ./data/train/ and evaluation dataset to ./data/val/ .

We only provide the ground truth images and the corresponding 4x downsampled LR images by [DUFVSR](https://github.com/yhjo09/VSR-DUF).

## Environment
  - Python (Tested on 3.6)
  - Tensorflow (Tested on 1.12.0)

## Training
We provide [pre-trained models](https://drive.google.com/file/d/1RuiuQngwRx0ea_ZTHXhbqIrLgfVCOoKD/view?usp=sharing), note that some models have been retrained and part of the codes have been modified, thus some methods may behave a little different from that reported in the paper.
Be free to use main.py to train any model you would like to.

## Testing
We provide [Vid4](https://drive.google.com/file/d/1-Sy3t0zgbUskX1rr2Vu7oM9ssLlfIvzd/view?usp=sharing) and [UDM10](https://drive.google.com/file/d/1IEURw2U4V9KNejw3YptPL6gWM2xLE6bq/view?usp=sharing) testing datasets.
It should be easy to use 'testvideo()' or 'testvideos()' functions for testing.

## Citation
If you find our code or datasets helpful, please consider citing our related works.
```
@inproceedings{PFNL,
  title={Progressive Fusion Video Super-Resolution Network via Exploiting Non-Local Spatio-Temporal Correlations},
  author={Yi, Peng and Wang, Zhongyuan and Jiang, Kui and Jiang, Junjun and Ma, Jiayi},
  booktitle={IEEE International Conference on Computer Vision (ICCV)},
  pages={3106-3115},
  year={2019},
}

@ARTICLE{wang2018mmcnn,
        author = {Wang, Zhongyuan and Yi, Peng and Jiang, Kui and Jiang, Junjun and Han, Zhen and Lu, Tao and Ma, Jiayi},
        journal={IEEE Transactions on Image Processing},
        title = {Multi-Memory Convolutional Neural Network for Video Super-Resolution},
        year={2018},
    }

@ARTICLE{MTUDM, 
author={Yi, Peng and Wang, Zhongyuan and Jiang, Kui and Shao, Zhenfeng and Ma, Jiayi}, 
journal={IEEE Transactions on Circuits and Systems for Video Technology}, 
title={Multi-Temporal Ultra Dense Memory Network For Video Super-Resolution}, 
year={2019}, 
doi={10.1109/TCSVT.2019.2925844}, 
ISSN={1051-8215},
}
```

## Contact
If you have questions or suggestions, please open an issue here or send an email to yipeng@whu.edu.cn.

## Visual Results
We show the visual results under 4x upscaling.
This frame is from auditorium in UDM10 testing dataset.

![Image text](https://github.com/psychopa4/PFNL/blob/master/pictures/comp0.jpg)

This frame is from photography in UDM10 testing dataset.

![Image text](https://github.com/psychopa4/PFNL/blob/master/pictures/comp1.jpg)

This is a real LR frame shoot by us.

![Image text](https://github.com/psychopa4/PFNL/blob/master/pictures/comp2.jpg)

## PSNR/SSIM on Vid4 test dataset (4xSR)
| Sequence | VESPCN | RVSR-LTD | MCResNet | DRVSR | FRVSR | DUF_52L | PFNL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|calendar | 22.20 / 0.7156 | 22.07 / 0.7041 | 22.44 / 0.7319 | 22.88 / 0.7586 | 23.46 / 0.7854 | 23.85 / 0.8052 | 24.37 / 0.8246 |
|city | 26.47 / 0.7246 | 26.44 / 0.7217 | 26.75 / 0.7454 | 27.06 / 0.7698 | 27.70 / 0.8099 | 27.97 / 0.8253 | 28.09 / 0.8385 |
|foliage | 25.07 / 0.6910 | 25.15 / 0.7004 | 25.30 / 0.7093 | 25.58 / 0.7307 | 25.96 / 0.7560 | 26.22 / 0.7646 | 26.51 / 0.7768 |
|walk | 28.40 / 0.8717 | 28.29 / 0.8677 | 28.76 / 0.8788 | 29.11 / 0.8876 | 29.69 / 0.8990 | 30.47 / 0.9118 | 30.64 / 0.9134 |
|average | 25.54 / 0.7507 | 25.49 / 0.7485 | 25.81 / 0.7664 | 26.16 / 0.7867 | 26.70 / 0.8126 | 27.13 / 0.8267 | 27.41 / 0.8383 |
|average* | 25.35 / 0.7557 | - / - | 25.45 / 0.7467 | 25.52 / 0.7600 | 26.69 / 0.8220 | 27.34 / 0.8327 | 27.41 / 0.8383 |

## PSNR/SSIM on UDM10 test dataset (4xSR)
| Sequence | VESPCN | RVSR-LTD | MCResNet | DRVSR | FRVSR | DUF_52L | PFNL |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
|archpeople | 35.37 / 0.9504 | 35.20 / 0.9485 | 35.46 / 0.9512 | 35.83 / 0.9547 | 36.24 / 0.9579 | 36.92 / 0.9638 | 38.35 / 0.9724 |
|archwall | 40.14 / 0.9581 | 39.80 / 0.9559 | 40.77 / 0.9637 | 41.16 / 0.9671 | 41.65 / 0.9710 | 42.53 / 0.9754 | 43.55 / 0.9792 |
|auditorium | 27.91 / 0.8837 | 27.49 / 0.8736 | 27.87 / 0.8874 | 29.00 / 0.9039 | 29.81 / 0.9181 | 30.27 / 0.9257 | 31.18 / 0.9369 |
|band | 33.55 / 0.9514 | 33.27 / 0.9481 | 33.88 / 0.9540 | 34.32 / 0.9579 | 34.54 / 0.9589 | 35.49 / 0.9660 | 36.01 / 0.9691 |
|caffe | 37.57 / 0.9647 | 37.22 / 0.9635 | 38.07 / 0.9676 | 39.08 / 0.9715 | 39.82 / 0.9746 | 41.03 / 0.9785 | 41.84 / 0.9808 |
|camera | 43.34 / 0.9886 | 43.36 / 0.9884 | 43.45 / 0.9887 | 45.19 / 0.9905 | 46.07 / 0.9912 | 47.30 / 0.9927 | 49.26 / 0.9941 |
|clap | 34.92 / 0.9544 | 34.57 / 0.9511 | 35.41 / 0.9578 | 36.20 / 0.9635 | 36.51 / 0.9659 | 37.70 / 0.9719 | 38.33 / 0.9756 |
|lake | 30.63 / 0.8255 | 30.69 / 0.8267 | 30.82 / 0.8323 | 31.15 / 0.8440 | 31.70 / 0.8623 | 32.06 / 0.8730 | 32.53 / 0.8865 |
|photography | 35.92 / 0.9581 | 35.61 / 0.9552 | 36.15 / 0.9594 | 36.60 / 0.9627 | 36.95 / 0.9655 | 38.02 / 0.9719 | 38.95 / 0.9768 |
|polyflow | 36.61 / 0.9489 | 36.43 / 0.9469 | 37.01 / 0.9521 | 37.91 / 0.9565 | 38.38 / 0.9597 | 39.25 / 0.9667 | 40.04 / 0.9734 |
|average | 35.60 / 0.9384 | 35.36 / 0.9358 | 35.89 / 0.9414 | 36.64 / 0.9472 | 37.17 / 0.9525 | 38.05 / 0.9586 | 39.00 / 0.9645 |