# Supervised Contrastive Learning for Scene Graph Generation


## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.

## Dataset

Check [DATASET.md](DATASET.md) for instructions of dataset preprocessing.

## Training **(IMPORTANT)**

### Prepare Faster-RCNN Detector
- You can download the pretrained Faster R-CNN we used in the paper: 
  - [VG](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EQIy64T-EK9Er9y8kVCDaukB79gJwfSsEIbey9g0Xag6lg?e=wkKHJs), 
  - [OIv6](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EfGXxc9byEtEnYFwd0xdlYEBcUuFXBjYxNUXVGkgc-jkfQ?e=lSlqnz), 
  - [OIv4](https://shanghaitecheducn-my.sharepoint.com/:u:/g/personal/lirj2_shanghaitech_edu_cn/EcxwkWxBqUdLuoP58vnUyMABR2-DC33NGj13Hcnw96kuXw?e=NveDcl) 
- put the checkpoint into the folder:
```
mkdir -p checkpoints/detection/pretrained_faster_rcnn/
# for VG
mv /path/vg_faster_det.pth checkpoints/detection/pretrained_faster_rcnn/
```

Then, you need to modify the pretrained weight parameter `MODEL.PRETRAINED_DETECTOR_CKPT` in configs yaml `configs/e2e_relBGNN_vg-oiv6-oiv4.yaml` to the path of corresponding pretrained rcnn weight to make sure you load the detection weight parameter correctly.


### Contrastive Learning
To apply contrastive learning representation to SGG models, you should train contrastive model.
We provide script for self supervised learning( in `script/contrastive_rel_train_self.sh`)
You don't need any other processed data. Just run that script then VGContrastive dataset automatically generate positive, negative samples and train contrastive model.

The result is saved in `checkpoint/contrastive_learning`
There is two main option about contrastive learning(MODEL.CONTRASTIVE.USE_CLUSTER or USE_OBJECT_INFO)
USE_CLUSTER mean using cluster information and USE_OBJECT_INFO mean get positive samples based on object label or predicate label. The experiment result in our paper is using predicate label.

### Apply pretrained contrastive model to baseline model
After contrastive learning, then you run script(`script/contrastive_rel_train_vg_predcls.sh`)
In that script, you should change the directory of contrastive model path.
Then everything is done!


## Test
Similarly, we also provide the `rel_test.sh` for directly produce the results from the checkpoint provide by us.
By replacing the parameter of `MODEL.WEIGHT` to the trained model weight and selected dataset name in `DATASETS.TEST`, you can directly eval the model on validation or test set.



## Acknowledgment
This repository is developed on top of the scene graph benchmarking framwork develped by [KaihuaTang](https://github.com/KaihuaTang/Scene-Graph-Benchmark.pytorch)
