# Show, Attend and Tell 
<b> This code is based on the code of [yunjey](https://github.com/yunjey/show-attend-and-tell)</b>, you can check the link in [[1]]][1] to know how to use this code.

## Some important modifications I made:

- Changed the image-encoder to Resnet-101 by Pytorch, *you may want to take a look at [prepro.py](prepro.py) and modify line 197 to change the encoder to other CNN models*.

- Fixed the evaluation code to get the right score on MSCOCO's offered validation set (yunjey pruned some sentences which don't satisfy some requirements, this leads to higher scores in evaluation phrase so we are not able to properly compare our model's performance to others).

- Added beam-search to the inference phrase, this is a modification version of this beam-search [[2]][2] algorithm, which improved the model's performance significantly.

## Reference:

[1]: https://github.com/yunjey/show-attend-and-tell "What up"

[2]: https://gist.github.com/nikitakit/6ab61a73b86c50ad88d409bac3c3d09f