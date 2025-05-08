<h1> Human Falling Detection and Tracking </h1>

Using Tiny-YOLO oneclass to detect each person in the frame and use
[AlphaPose](https://github.com/MVIG-SJTU/AlphaPose) to get skeleton-pose and then use
[ST-GCN](https://github.com/yysijie/st-gcn) model to predict action from every 30 frames
of each person tracks.

Which now support 7 actions: Standing, Walking, Sitting, Lying Down, Stand up, Sit down, Fall Down.

<div align="center">
    <img src="sample1.gif" width="416">
</div>

## Prerequisites

-   Python > 3.6
-   Pytorch > 1.3.1

Original test run on: i7-8750H CPU @ 2.20GHz x12, GeForce RTX 2070 8GB, CUDA 10.2

## Data

This project has trained a new Tiny-YOLO oneclass model to detect only person objects and to reducing
model size. Train with rotation augmented [COCO](http://cocodataset.org/#home) person keypoints dataset
for more robust person detection in a variant of angle pose.

For actions recognition used data from [Le2i](http://le2i.cnrs.fr/Fall-detection-Dataset?lang=fr)
Fall detection Dataset (Coffee room, Home) extract skeleton-pose by AlphaPose and labeled each action
frames by hand for training ST-GCN model.

## Basic Use


## Reference

-   AlphaPose : https://github.com/Amanbhandula/AlphaPose
-   ST-GCN : https://github.com/yysijie/st-gcn
