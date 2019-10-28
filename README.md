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

Unzip the training dataset to ./data/train/ and evaluation dataset to ./data/val/ .

We only provide the ground truth images and the corresponding 4x downsampled LR images by [DUFVSR](https://github.com/yhjo09/VSR-DUF).

## Environment
  - Python (Tested on 3.6)
  - Tensorflow (Tested on 1.12.0)

## Training
We provide [pre-trained models](https://drive.google.com/file/d/1RuiuQngwRx0ea_ZTHXhbqIrLgfVCOoKD/view?usp=sharing), note that some models have been retrained, and they behave a little different from that reported in the papers.
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