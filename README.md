# LaneSegNet
This repo is the PyTorch implementation for LaneSegNet:

We propose LaneSegNet, a robust lane detection method that introduces a progressive lane reconstruction pipeline from lane points to lane segments, ultimately forming complete lanes. Specifically, lane points and multiple reference points are initially extracted and represented as hierarchical graph nodes. A graph enhancement module is utilized to capture both global and local structural relationships between reference points and lane points, respectively. Then a lane segment is constructed by associating adjacent lane points with each reference point. Multiple lane segments will be utilized to reconstruct a complete lane.   LaneSegNet can efficiently establish the topological relationships of sparse lane points across different spatial scales and robustly detect lanes in challenging situations.  


## Installation
 1. Create a conda virtual environment and activate it.
    ```shell
    conda create -n lanesegnet python=3.7 -y
    conda activate lanesegnet
    conda install pytorch==1.6.0 torchvision==0.7.0 cudatoolkit=10.1 -c pytorch -y
    pip install -r requirements/build.txt
    ```
 2. Clone this repository and enter it:
    ```Shell
    git clone https://github.com/ustclbh/LaneSegNet.git
    cd LaneSegNet
    python setup.py develop
    ```

## Dataset
Download datasets and put it into `[data-path]` folder. And edit the `data_root` in the config file to your dataset path.
### CULane
[\[CULane website\]](https://xingangpan.github.io/projects/CULane.html)
[\[Download\]](https://drive.google.com/drive/folders/1mSLgwVTiaUMAb4AVOWwlCD5JcWdrwpvu)


The directory structure should be like follows:
```
[data-path]/culane
├── driver_23_30frame
├── driver_37_30frame
├── driver_100_30frame
├── driver_161_90frame
├── driver_182_30frame
├── driver_193_90frame
└── list
    └── test_split
    |   ├── test0_normal.txt
    |   ├── test1_crowd.txt
    |   ├── test2_hlight.txt
    |   ├── test3_shadow.txt
    |   ├── test4_noline.txt
    |   ├── test5_arrow.txt
    |   ├── test6_curve.txt
    |   ├── test7_cross.txt
    |   └── test8_night.txt
    └── train.txt
    └── train_gt.txt
    └── test.txt
    └── test_gt.txt
    └── val.txt

```
### TuSimple
[\[TuSimple website\]](https://github.com/TuSimple/tusimple-benchmark/tree/master/doc/lane_detection)
[\[Download\]](https://github.com/TuSimple/tusimple-benchmark/issues/3)

The directory structure should be like follows:
```
[data-path]/tusimple
├── clips
├── label_data_0313.json
├── label_data_0531.json
├── label_data_0601.json
├── label_data_0601.json
├── test_label.json
└── test_baseline.json

```

## Training
To train the model,  run the following commands.
```shell
cd tools
sh dist_train.sh culane final_exp_res18_s8 ./output
```





