# SNE-RoadSeg


## Introduction
This is the official PyTorch implementation of [**SNE-RoadSeg: Incorporating Surface Normal Information into Semantic Segmentation for Accurate Freespace Detection**](http://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123750341.pdf), accepted by [ECCV 2020](https://eccv2020.eu/). This is our [project page](https://sites.google.com/view/sne-roadseg).

In this repo, we provide the training and testing setup for the [KITTI Road Dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). We test our code in Python 3.7, CUDA 10.0, cuDNN 7 and PyTorch 1.1. We provide `Dockerfile` to build the docker image we use.

<p align="center">
<img src="doc/kitti.gif" width="100%"/>
</p>

<p align="center">
<img src="doc/sne-roadseg.png" width="100%"/>
</p>


## Setup
Please setup the KITTI Road Dataset and pretrained weights according to the following folder structure:
```
SNE-RoadSeg
 |-- checkpoints
 |  |-- kitti
 |  |  |-- kitti_net_RoadSeg.pth
 |-- data
 |-- datasets
 |  |-- kitti
 |  |  |-- training
 |  |  |  |-- calib
 |  |  |  |-- depth_u16
 |  |  |  |-- gt_image_2
 |  |  |  |-- image_2
 |  |  |-- validation
 |  |  |  |-- calib
 |  |  |  |-- depth_u16
 |  |  |  |-- gt_image_2
 |  |  |  |-- image_2
 |  |  |-- testing
 |  |  |  |-- calib
 |  |  |  |-- depth_u16
 |  |  |  |-- image_2
 |-- examples
 ...
```
`image_2`, `gt_image_2` and `calib` can be downloaded from the [KITTI Road Dataset](http://www.cvlibs.net/datasets/kitti/eval_road.php). We implement `depth_u16` based on the LiDAR data provided in the KITTI Road Dataset, and it can be downloaded from [here](https://drive.google.com/file/d/16ft3_V8bMM-av5khZpP6_-Vtz-kF6X7c/view?usp=sharing). Note that `depth_u16` has the `uint16` data format, and the real depth in meters can be obtained by ``double(depth_u16)/1000``. Moreover, the pretrained weights `kitti_net_RoadSeg.pth` for our SNE-RoadSeg-152 can be downloaded from [here](https://drive.google.com/file/d/17eqDPTs0Sv83Q7e-IWox_69bCbiXUzPU/view?usp=sharing).


## Usage

### Run an example ###
We provide one example in `examples`. To run it, you only need to setup the `checkpoints` folder as mentioned above. Then, run the following script:
```
bash ./scripts/run_example.sh
```
and you will see `normal.png`, `pred.png` and `prob_map.png` in `examples`. `normal.png` is the normal estimation by our SNE; `pred.png` is the freespace prediction by our SNE-RoadSeg; and `prob_map.png` is the probability map predicted by our SNE-RoadSeg.

### Testing for KITTI submission
For KITTI submission, you need to setup the `checkpoints` and the `datasets/kitti/testing` folder as mentioned above. Then, run the following script:
```
bash ./scripts/test.sh
```
and you will get the prediction results in `testresults`. After that you can follow the [submission instructions](http://www.cvlibs.net/datasets/kitti/eval_road.php) to transform the prediction results into the BEV perspective for submission.

If everything works fine, you will get a MaxF score of **96.74** for **URBAN**. Note that this is our re-implemented weights, and it is very similar to the reported ones in the paper (a MaxF score of **96.75** for **URBAN**).

### Training on the KITTI dataset
For training, you need to setup the `datasets/kitti` folder as mentioned above. You can split the original training set into a new training set and a validation set as you like. Then, run the following script:
```
bash ./scripts/train.sh
```
and the weights will be saved in `checkpoints` and the tensorboard record containing the loss curves as well as the performance on the validation set will be save in `runs`. Note that `use-sne` in `train.sh` controls if we will use our SNE model, and the default is True. If you delete it, our RoadSeg will take depth images as input, and you also need to delete `use-sne` in `test.sh` to avoid errors when testing.



## Citation
If you use this code for your research, please cite our paper.
```
@inproceedings{fan2020sne,
  author = {Fan, Rui and Wang, Hengli and Cai, Peide and Liu, Ming},
  title = {SNE-RoadSeg: Incorporating Surface Normal Information into Semantic Segmentation for Accurate Freespace Detection},
  booktitle = {Proceedings of the European Conference on Computer Vision (ECCV)},
  year = {2020},
  organization = {Springer},
}
```


## Acknowledgement
Our code is inspired by [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix), and we thank [Jun-Yan Zhu](https://github.com/junyanz) for their great work.
