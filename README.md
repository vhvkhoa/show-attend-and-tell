# Show, Attend and Tell 
<b> This code is based on the code of a github user [yunjey](https://github.com/yunjey/show-attend-and-tell)</b>. It is an attempt to reproduce the performance of the image captioning method proposed in [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf).

## Some important modifications I made:

- Changed the image-encoder to Resnet-101 by Pytorch, *you may want to take a look at [prepro.py](prepro.py) and modify line 197 to change the encoder to other CNN models*.

- Fixed the evaluation code to get the right score on [MSCOCO dataset](http://cocodataset.org)'s offered validation set *(the [yunjey](https://github.com/yunjey/show-attend-and-tell)'s code pruned some sentences which don't satisfy some requirements, this leads to higher scores in evaluation phrase so we are not able to properly compare our model's performance to others)*.

- Added beam-search to the inference phrase, this is a modification version of this [beam search](https://gist.github.com/nikitakit/6ab61a73b86c50ad88d409bac3c3d09f) algorithm, which improved the model's performance significantly compared to other versions of beam search.

## Dependencies:

- Python 2.7
- tensorflow 1.4 **Higher versions are currently not able to run this code, we are trying to fix this**
- pytorch
- torchvision
- skimage
- tqdm
- pandas

## Getting Started:

If you want to train or evaluate the models, you need to clone the repo of [pycocoevalcap](https://github.com/tylin/coco-caption), run this line in your $HOME directory containing this repo:

> git clone https://github.com/tylin/coco-caption.git

### Training:

In order to train the model, you need to download the dataset's images and annotations, run this line to download [MSCOCO](http://cocodataset.org) 2017 version's training, validation sets and 2014 testing set:

> bash download.sh

After that, preprocess the images and captions and then run training code, you're free to set configuration of the training phrase by modifying [train.py](train.py) file:

> python prepro.py
> python train.py

Make sure that you have enough space in your drive, it would take about **125GB** after preprocessing.

While training, you can observe the process by tensorboard:

> tensorboard --logdir=log/

### Evaluation and Inference:

## References:

The code we used as the backbone of our code: https://github.com/yunjey/show-attend-and-tell

The code of beam-search: https://gist.github.com/nikitakit/6ab61a73b86c50ad88d409bac3c3d09f

The authors' code: https://github.com/kelvinxu/arctic-captions 