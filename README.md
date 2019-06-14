# Project-2-Let Us be a Naturalist
## Introduction
This repository is about the second project **Let Us be a Naturalist** of AI for Mechatronics (MCEN90048), which mainly investigate the fine-grained image classification for iNaturalist 2019 FGVC6 at CVPR 2019.

The codes is modified based on 'Week 11 - A Working Example for Project 2 on Local Machine' created by Richard Wang .

The codes is able to build three CNN models, including InceptionV3, Xception and InceptionResNetV2 using Keras and train them according to iNaturalist 2019 dataset. Additionally, the required submission CSV file can also be generated according to the fine-trained model. 

## Note
1. In training mode, users can choose to train the three model from scratch or load pre-trained model to continue training. The base weights for InceptionV3, Xception and InceptionResNetV2 are pre-trained on ImageNet.

2. If users want to continue to train the model or get corresbonding CSV file, please make sure the desired weight is put in the ckpt_folder, whose path is define in Assignment2_main.py **ckpt_folder = FLAGS.DEFAULT_OUT + 'trial_{}_{}_{}/'.format(image_size, target_size, drop_out)**.
