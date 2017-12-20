# clcNet Torch Implementation

This is the Torch implementation of clcNet, an efficient convolutional neural network, presented in [clcNet: Improving the Efficiency of Convolutional Neural Network using Channel Local Convolutions](https://arxiv.org/abs/1712.06145)

The code is modified from [fb.resnet.torch](https://github.com/facebook/fb.resnet.torch).

## Training and Validation

1. Install Torch, cuDNN on a machine with CUDA GPU. You can refer  [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md) for a step-by-step guide.

2. Download the [ImageNet](http://image-net.org/download-images) dataset and prepare the data. Please refer [here](https://github.com/facebook/fb.resnet.torch/blob/master/INSTALL.md#download-the-imagenet-dataset) for the instructions of data preparation.

3. Clone this repo: ```git clone https://github.com/dqzhang17/clcnet.torch.git```

4. For training from scratch :
Choose clcneta or clcnetb for netType, and change nGPU, batchSize etc. according to your hardware infrastructure:
```bash
th main.lua -nGPU 4 -batchSize 256 -nThreads 8 -data [your_imagenet_folder] -netType clcneta -nEpochs 100 -LR 0.1 -shareGradInput true
```
Or you can build your own custom clcnet by choosing layer counts for different stages. For instance, let a=1, b=1, c=5, d=2:
```bash
th main.lua -nGPU 4 -batchSize 256 -nThreads 8 -data [your_imagenet_folder] -a 1 -b 1 -c 5 -d 2 -nEpochs 100 -LR 0.1 -shareGradInput true
```

5. For validation :
```bash
th main.lua -nGPU 4 -batchSize 256 -nThreads 8 -data [your_imagenet_folder] -retrain [your_trained_model.t7] -testOnly true
```

## Results on ImageNet and Pretrained Models


The accuracy rates shown here are 224x224 single-model single-crop test accuracy on the ImageNet-1k validation dataset.

| Network       | Layer Config (a,b,c,d) | Top-1 Accuracy | Torch Model |
| ------------- | ----------- | ----------- | ----------- |
| clcNet-A  |(1, 1, 5, 2) | 70.4%    | [Download (26.2MB)](https://drive.google.com/file/d/1HTMHSeYx0JHlYbwevzuWG5rW8umCoon0/view?usp=sharing)       |
| clcNet-B  |(1, 1, 7, 3) |71.6%     | [Download (32.9MB)](https://drive.google.com/file/d/1nxWnmrtg1m6Hm_0Tgby3v9WQCBzjrSMI/view?usp=sharing)       |


### BibText

	@inproceedings{zhang2017clcnet,
	  title={clcNet: Improving the Efficiency of Convolutional Neural Network using Channel Local Convolutions},
	  author={Zhang, Dong-Qing},
	  journal={arXiv preprint arXiv:1712.06145},
	  year={2017}
	}
