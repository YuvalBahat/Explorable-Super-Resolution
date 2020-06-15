The official PyTorch implementation of the paper "Explorable Super Resolution" by Yuval Bahat and Tomer Michaeli (CVPR 2020).

## Table of Contents
1. [Overview?](#overview)
1. [Dependencies](#dependencies)
1. [Acknowledgement](#Acknowledgement)
1. [Running the GUI](#GUI_run)
1. [Exploring with the GUI](#GUI_usage)
1. [Training an explorable super-resolution network](#Training)
1. [Using the consistency enforcing module (CEM) for other purposes](./CEM)

## Overview 
### This repository includes:<a name="repository_includes"></a>
1. Code for a **Graphical User Interface (GUI)** allwoing a user to perform explorable super resoution and edit a low-resoultion image in real time. Pre-trained backend models are available for download. 
1. Code for **training an explorable super resolution model** yourself. This model can then be used to replace the available pre-trained models as the GUI backend.
1. Implementation of the **Consistency Enforcing Module (CEM)** that can wrap any existing (and even pre-trained) super resolution network, modifying its high-resolution outputs to be consistent with the low-resolution input.

The overall explorable super resolution framework is shown in the figure below. It consists of a super-resolution neural network, a consistency enforcing module (CEM) and a graphical user interface (GUI). 
<p align="center">
   <img src="fig_framework_scheme_4_github.png">
</p>

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

## Exploring using our GUI<a name="GUI_usage"></a>

## Training the backend exploration network<a name="Training"></a>
