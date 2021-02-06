An official PyTorch implementation of "Explorable Super Resolution" by Yuval Bahat and Tomer Michaeli (CVPR 2020).

## Table of Contents
1. [Overview](#overview)
1. [Dependencies](#dependencies)
1. [Acknowledgement](#acknowledgement)
1. [Running the GUI](#GUI_run)
1. [Exploring with the GUI](#GUI_usage)
1. [Training an explorable super-resolution network](#Training)
1. [Using the consistency enforcing module (CEM) for other purposes](./CEM)

## Overview 
The overall explorable super resolution framework is shown in the figure below. It consists of a super-resolution neural network, a consistency enforcing module (CEM) and a graphical user interface (GUI). 
<p align="center">
   <img src="fig_framework_scheme_4_github.png">
</p>

### This repository includes:<a name="repository_includes"></a>
1. Code for a **Graphical User Interface (GUI)** allwoing a user to perform explorable super resoution and edit a low-resoultion image in real time. Pre-trained backend models for the 4x case are available for download, though our method supports any integer super-resolution factor.

1. Code for **training an explorable super resolution model** yourself. This model can then be used to replace the available pre-trained models as the GUI backend.
1. Implementation of the **Consistency Enforcing Module (CEM)** that can wrap any existing (and even pre-trained) super resolution network, modifying its high-resolution outputs to be consistent with the low-resolution input.


You can run our **GUI** and use its tools to explore the abundant different high-resolution images matching an input low-resolution image. The backend of this GUI comprises an explorable super-resolution netwrok. You can either download a pre-trained model, or you can **train a model by yourself**. Finally, our consistency enforcing module (CEM) can be used as a standalone component, to **wrap any super-resolution model**, whether before or after its training, for guranteeing the consistency of its outputs.

Our CEM assumes the default bicubic downsampling kernel, but in needs access to the actual downsampling kernel corresponding to the low-resolution image, in order to guarantee the consistency of our framework's outputs. To this end, GUI users can utilize the incorporated [KernelGAN](http://www.wisdom.weizmann.ac.il/~vision/kernelgan/) kernel estimation method by Bell-Kligler et al., which may improve consistency in some cases.

## Dependencies

- Python >= 3.6
- [PyTorch >= 1.1.0](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)
<!--- Other Python packages: `pip install numpy opencv-python lmdb`-->

## Acknowledgement
Code architecture is based on an older version of [BasicSR](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix).

## Running the explorable SR GUI<a name="GUI_run"></a>
1. *Train or download a pre-trained explorable SR model:*  
Our GUI enables exploration by utilizing a backend explorable SR network. Therefore to run it, you first need to either [train](#Training) or [download a pre-trained](https://drive.google.com/file/d/1UmF0Dy_c97CMiyMFG16goJxzXXwpQOUG/view?usp=sharing) model. The corresponding [pre-trained discriminator is available here](https://drive.google.com/file/d/1VsDX6dhQyszW1Lq3wUp5I19mBhuKg9O2/view?usp=sharing), in case you want to fine-tune the model.
1. *(Optional) Download a pre-trained ESRGAN model:*  
Download a [pre-trained ESRGAN model](https://drive.google.com/file/d/1bWeH3zo0OIoCYUjei2pkCBm-ATlkwhpK/view?usp=sharing), to display the (single) super-resolved output by the state-of-the-art [ESRGAN](https://arxiv.org/abs/1809.00219) method.
1. *Update paths:*  
Update the necessary fields in the [`GUI_SR.json`](./options/test/GUI_SR.json) file.
1. *Run the GUI:*  
   ```
   python GUI.py -opt ./options/test/GUI_SR.json  
   ```

## Exploring using our GUI<a name="GUI_usage"></a>
I hope to add here a full description of all our GUI exploration tools soon. In the meantime, please refer to the description in appendix D of [our paper](https://drive.google.com/file/d/1N6pwutE_wxx8xDx29zvItjDdqO-CLklG/view?usp=sharing).

## Training the backend exploration network<a name="Training"></a>
1. *Download training set:*  
Download a dataset of high-resolution training images. We used the training subset of the [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) dataset.
1. *Training-set preparation* (Requires updating the 'dataset_root_path' field in each of the scripts below):
   1. Create image crops of your high resolution (HR) image training set using [`extract_subimgs_single.py`](./scripts/extract_subimgs_single.py).
   1. Create two new folders containing pairs of corresponding HR and LR image crops, using [`generate_mod_LR_bic.py`](./scripts/generate_mod_LR_bic.py).
   1. Create two corresponding `lmdb` files using [`create_lmdb.py`](./scripts/create_lmdb.py) (change the 'HR_images' flag for the LR file).
1. *Download initialization model:*  
Download a [pre-trained ESRGAN model](https://drive.google.com/file/d/1bWeH3zo0OIoCYUjei2pkCBm-ATlkwhpK/view?usp=sharing) for weights initialization (This model is for a 4x super-resolution. Other factors require a different model).
1. *Update parameters:*  
Update the necessary (and optionally other) fields in the [`train_explorable_SR.json`](./options/train/train_explorable_SR.json) file.
1. *Train the model:*  
   ```
   python train.py -opt ./options/train/train_explorable_SR.json  
   ```
