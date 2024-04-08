# Theoretically Achieving Continuous Representation of Oriented Bounding Boxes
## Abstract
Considerable efforts have been devoted to Oriented Object Detection (OOD). However, one lasting issue regarding the discontinuity in Oriented Bounding Box (OBB) representation remains unresolved, which is an inherent bottleneck for extant OOD methods. This paper endeavors to completely solve this issue in a theoretically guaranteed manner and puts an end to the ad-hoc efforts in this direction. Prior studies typically can only address one of the two cases of discontinuity: rotation and aspect ratio, and often inadvertently introduce decoding discontinuity, e.g. Decoding Incompleteness (DI) and Decoding Ambiguity (DA) as discussed in literature. Specifically, we propose a novel representation method called **C**ontinuous **OBB** (**COBB**), which can be readily integrated into existing detectors e.g. Faster-RCNN as a plugin. It can theoretically ensure continuity in bounding box regression which to our best knowledge, has not been achieved in literature for rectangle-based object representation. For fairness and transparency of experiments, we have developed a modularized benchmark based on the open-source deep learning framework Jittor's detection toolbox JDet for OOD evaluation. On the popular DOTA dataset, by integrating Faster-RCNN as the same baseline model, our new method outperforms the peer method Gliding Vertex by 1.13\% mAP<sub>50</sub> (relative improvement 1.54\%), and 2.46\% mAP<sub>75</sub> (relative improvement 5.91\%), without any tricks.

(accepted by CVPR 2024)

## Install
```shell
git clone https://github.com/514flowey/JDet-cobb.git
cd JDet
python -m pip install -r requirements.txt
python setup.py develop
```
It's suggested to run
```shell
JITTOR_HOME=build/ python -m jittor_utils.install_cuda
```
to install cuda for jittor.

Please refer to [JDet](https://github.com/Jittor/JDet) for more information.


## Getting Started

### Datasets
The following datasets are supported in JDet, please check the corresponding document before use. 

DOTA1.0/DOTA1.5/DOTA2.0 Dataset: [dota.md](docs/dota.md).

You can also build your own dataset by convert your datas to DOTA format.

### Config
JDet defines the used model, dataset and training/testing method by `config-file`, please check the [config.md](docs/config.md) to learn how it works.
### Train
```shell
CUDA_VISIBLE_DEVICES=0 JITTOR_HOME=build/ python tools/run_net.py \
--config-file configs/faster_rcnn/faster_rcnn_obb_r50_fpn_1x_dota.py
```

### Test
If you want to test the downloaded trained models, please use ```--load_from {you_checkpointspath}```.
```shell
CUDA_VISIBLE_DEVICES=0 JITTOR_HOME=build/ python tools/run_net.py \
 --config-file configs/roi_transformer/roi_transformer_obb_r50_fpn_1x_dota.py \
 --load_from weights/checkpoints/roi_transformer_obb_r50_fpn_1x_dota_ckpt12.pkl \
 --task test
```

## Models

|         Models         | Dataset  | Sub_Image_Size/Overlap  |   Train Aug     | Test Aug  |  mAP   |                                           Config                                            |           Download          |
|:-------------:|:--------:|:-----------------------:|:---------------:|:---------:|:------:|:-------------------------------------------------------------------------------------------:|:---------------------------------:|
|     Faster R-CNN     | DOTA1.0  |        1024/200         |      flip       |     -     | 73.01  |                     [config](configs/benchmark/faster_rcnn/faster_rcnn_obb_r50_fpn_1x_dota.py)                      |       [Tsinghua](https://cloud.tsinghua.edu.cn/f/5d5c30ec6bc34941a515/?dl=1) <br> [Baidu Disk](https://pan.baidu.com/s/1i4fhG1WBfTSHE6GKoRSG-g?pwd=cobb)        |
|     Faster R-CNN <br> + COBB-sig     | DOTA1.0  |        1024/200         |      flip       |     -     | 74.00  |                     [config](configs/benchmark/cobb/faster_rcnn_cobb_sig_r50_fpn_1x_dota.py)                      |       [Tsinghua](https://cloud.tsinghua.edu.cn/f/e1937025f9c842e9bb10/?dl=1) <br> [Baidu Disk](https://pan.baidu.com/s/1R3-qKLAY98xpD0If8dtAIQ?pwd=cobb)        |
|     Faster R-CNN <br> + COBB-ln     | DOTA1.0  |        1024/200         |      flip       |     -     | 74.44  |                     [config](configs/benchmark/cobb/faster_rcnn_cobb_ln_r50_fpn_1x_dota.py)                      |       [Tsinghua](https://cloud.tsinghua.edu.cn/f/95c228b9239b4eee8675/?dl=1) <br> [Baidu Disk](https://pan.baidu.com/s/1t62KESWd4rpFsqpQzAokLQ?pwd=cobb)        |
|     RoI Transformer     | DOTA1.0  |        1024/200         |      flip       |     -     | 75.59  |                     [config](configs/benchmark/roi_transformer/roi_transformer_obb_r50_fpn_1x_dota.py)                      |       [Tsinghua](https://cloud.tsinghua.edu.cn/f/fbf410ed17b348e69497/?dl=1) <br> [Baidu Disk](https://pan.baidu.com/s/18bw0W-xcuhMKq3nImE7wxQ?pwd=cobb)        |
|     RoI Transformer <br> + COBB-ln-sig     | DOTA1.0  |        1024/200         |      flip       |     -     | 76.55  |                     [config](configs/benchmark/cobb/roi_transformer_cobb_ln_sig_r50_fpn_1x_dota.py)                      |       [Tsinghua](https://cloud.tsinghua.edu.cn/f/6d4453e2f0bf45d18523/?dl=1) <br> [Baidu Disk](https://pan.baidu.com/s/1Erz6KBLTJwVd-eeeZiD8Cw?pwd=cobb)        |
|     RoI Transformer <br> + COBB-ln-ln     | DOTA1.0  |        1024/200         |      flip       |     -     | 76.53  |                     [config](configs/benchmark/cobb/roi_transformer_cobb_ln_ln_r50_fpn_1x_dota.py)                      |       [Tsinghua](https://cloud.tsinghua.edu.cn/f/c41db9e4c72142fdad3a/?dl=1) <br> [Baidu Disk](https://pan.baidu.com/s/111vgoOowjmWDXZhx_kVpGA?pwd=cobb)        |

## Contact Us

Email: xzk23@mails.tsinghua.edu.cn

If there is any problem about JDet or Jittor, please refer to [Jittor](https://github.com/Jittor/jittor) and [JDet](https://github.com/Jittor/JDet)


## Citation


```
@article{xiao2024theoretically,
  title={Theoretically Achieving Continuous Representation of Oriented Bounding Boxes},
  author={Xiao, Zikai and Yang, Guo-Ye and Yang, Xue and Mu, Tai-Jiang and Yan, Junchi and Hu, Shi-min},
  journal={arXiv preprint arXiv:2402.18975},
  year={2024}
}
```

## Reference
1. [Jittor](https://github.com/Jittor/jittor)
2. [JDet](https://github.com/Jittor/JDet)
3. [mmrotate](https://github.com/open-mmlab/mmrotate)


