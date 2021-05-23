##   cs224u Project

OP-GAN from: https://github.com/tohinz/semantic-object-accuracy-for-generative-text-to-image-synthesis

## Setting Up CLIP
First, [install PyTorch 1.7.1](https://pytorch.org/get-started/locally/) and torchvision, as well as small additional dependencies, and then install this repo as a Python package. On a CUDA GPU machine, the following will do the trick:

```bash
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm
$ pip install git+https://github.com/openai/CLIP.git
```

Replace `cudatoolkit=11.0` above with the appropriate CUDA version on your machine or `cpuonly` when installing on a machine without a GPU.

Run the below command to ensure clip is setup correctly:
```
python clip_demo.py
```


## Use Our Model (OP-GAN)
#### Dependencies
- python 3.8.5
- pytorch 1.7.1

Go to ``OP-GAN``.
Please add the project folder to PYTHONPATH and install the required dependencies:

```
conda env create -f environment.yml
```

#### Data
- MS-COCO:
    - [download](https://www2.informatik.uni-hamburg.de/wtm/software/semantic-object-accuracy/data.tar.gz) our preprocessed data (bounding boxes, bounding box labels, preprocessed captions), save to `data/` and extract
        - the preprocessed captions are obtained from and are the same as in the [AttnGAN implementation](https://github.com/taoxugit/AttnGAN)
        - the generateod bounding boxes for evaluating at test time were generated with code from the [Obj-GAN](https://github.com/jamesli1618/Obj-GAN)
    - obtain the train and validation images from the 2014 split [here](http://cocodataset.org/#download), extract and save them in `data/train/` and `data/test/`
    - download the pre-trained DAMSM for COCO model from [here](https://github.com/taoxugit/AttnGAN), put it into `models/` and extract

#### Training
- to start training run `sh train.sh gpu-ids` where you choose which gpus to train on
    - e.g. `sh train.sh 0,1,2,3`
- training parameters can be adapted via `code/cfg/dataset_train.yml`, if you train on more/fewer GPUs or have more VRAM adjust the batch sizes as needed
- make sure the DATA_DIR in the respective `code/cfg/cfg_file_train.yml` points to the correct path
- results are stored in `output/`

#### Evaluating
- update the eval cfg file in `code/cfg/dataset_eval.yml` and adapt the path of `NET_G` to point to the model you want to use (default path is to the pretrained model linked below)
- run `sh sample.sh gpu-ids` to generate images using the specified model
    - e.g. `sh sample.sh 0`

#### Pretrained Models
- OP-GAN: [download](https://www2.informatik.uni-hamburg.de/wtm/software/semantic-object-accuracy/op-gan.pth) and save to `models`


## Acknowledgement
- Code and preprocessed metadata for the experiments on MS-COCO are adapted from [AttnGAN](https://github.com/taoxugit/AttnGAN) and [AttnGAN+OP](https://github.com/tohinz/multiple-objects-gan).
- Code to generate bounding boxes for evaluation at test time is from the [Obj-GAN](https://github.com/jamesli1618/Obj-GAN) implementation.
- Code for using YOLOv3 is adapted from [here](https://pjreddie.com/darknet/), [here](https://github.com/eriklindernoren/PyTorch-YOLOv3), and [here](https://github.com/ayooshkathuria/pytorch-yolo-v3).

## Citing
If you find our model useful in your research please consider citing:

```
@article{hinz2019semantic,
title     = {Semantic Object Accuracy for Generative Text-to-Image Synthesis},
author    = {Tobias Hinz and Stefan Heinrich and Stefan Wermter},
journal   = {arXiv preprint arXiv:1910.13321},
year      = {2019},
}
```
