<p style="font-size:30px" align="center">
   <a href="https://sites.google.com/view/yuval-bahat/home">Yuval Bahat</a> &ensp; & &ensp;<a href="https://tomer.net.technion.ac.il/">Tomer Michaeli</a>
   <br><br>
   <img src="7173-teaser.gif" height="400">
</p>

## Abstract
Single image super resolution (SR) has seen major performance leaps in recent years. However, existing methods do not allow exploring the infinitely many plausible reconstructions that might have given rise to the observed low-resolution (LR) image. These different explanations to the LR image may dramatically vary in their textures and fine details, and may often encode completely different semantic information. In this work, we introduce the task of explorable super resolution. We propose a framework comprising a graphical user interface with a neural network backend, allowing editing the SR output so as to explore the abundance of plausible HR explanations to the LR input. At the heart of our method is a novel module that can wrap any existing SR network, analytically guaranteeing that its SR outputs would precisely match the LR input, when downsampled. Besides its importance in our setting, this module is guaranteed to decrease the reconstruction error of any SR network it wraps, and can be used to cope with blur kernels that are different from the one the network was trained for. We illustrate our approach in a variety of use cases, ranging from medical imaging and forensics, to graphics.

## Try it yourself :mag:
Use our GUI to explore the infinite high-resolution images corresponding to an input low-resolution image. You can use our pre-trained explorable super-resolution model, or train one yourself. You can also utilize our CEM to enforce consistency on any super resolution model, regardless of explorability. Please find details in our [code repository](./codes).

## Resources
1. [Paper](https://drive.google.com/file/d/1N6pwutE_wxx8xDx29zvItjDdqO-CLklG/view?usp=sharing) (Including supplementary material)
1. [Short oral presentation](https://youtu.be/OaksbqfP1rY) (5 mins, CVPR 2020)
1. [Slides](https://drive.google.com/file/d/134Lr4G4ffr8A93m24iVkaW8DeYH1AHss/view?usp=sharing)
1. [Longer talk](https://youtu.be/sT8qeRpTetk) (47 mins, MIT vision seminar)
1. [Code](https://github.com/YuvalBahat/Explorable-Super-Resolution/tree/gh-pages/codes)

### BibTex
    @inproceedings{bahat2020explorable,
      title={Explorable Super Resolution},
      author={Bahat, Yuval and Michaeli, Tomer},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={2716--2725},
      year={2020}
    }
    
