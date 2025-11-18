<div id="top"></div>
<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->

<!-- PROJECT SHIELDS -->

<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
<!-- 
***[![MIT License][license-shield]][license-url]
-->

<!-- GETTING STARTED -->

## Getting Started

This is an example of how to set up USB locally.
To get a local copy up, running follow these simple example steps.

### Prerequisites

USB is built on pytorch, with torchvision, torchaudio, and transformers.

To install the required packages, you can create a conda environment:

```sh
conda create --name usb python=3.8
```

then use pip to install required packages:

```sh
pip install -r requirements.txt
```

From now on, you can start use USB by typing 

```sh
python train.py --c config/usb_cv/fixmatch/fixmatch_cifar100_200_0.yaml
```

### Installation

We provide a Python package *semilearn* of USB for users who want to start training/testing the supported SSL algorithms on their data quickly:

```sh
pip install semilearn
```

<p align="right">(<a href="#top">back to top</a>)</p>

### Development

You can also develop your own SSL algorithm and evaluate it by cloning USB:

```sh
git clone https://github.com/microsoft/Semi-supervised-learning.git
```

<p align="right">(<a href="#top">back to top</a>)</p>


### Prepare Datasets

The detailed instructions for downloading and processing are shown in [Dataset Download](./preprocess/). Please follow it to download datasets before running or developing algorithms.

<p align="right">(<a href="#top">back to top</a>)</p>


<!-- USAGE EXAMPLES -->

## Usage

USB is easy to use and extend. Going through the bellowing examples will help you familiar with USB for quick use, evaluate an existing SSL algorithm on your own dataset, or developing new SSL algorithms.

### Quick Start with USB package

<!-- TODO: add quick start example and refer lighting notebook -->

Please see [Installation](#installation) to install USB first. We provide colab tutorials for:

- [Beginning example](https://colab.research.google.com/drive/1lFygK31jWyTH88ktao6Ow-5nny5-B7v5)
- [Customize datasets](https://colab.research.google.com/drive/1zbswPm1sM8j0fndUQOeqX2HADdYq-wOw)

### Start with Conda

**Step1: Create your environment**



**Step2: Clone the project**



### Start with Docker

**Step1: Check your environment**

You need to properly install Docker and nvidia driver first. To use GPU in a docker container
You also need to install nvidia-docker2 ([Installation Guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker)).
Then, Please check your CUDA version via `nvidia-smi`

**Step2: Clone the project**

```shell
git clone https://github.com/Zheng-freshman/ilfhctn.git
```

**Step3: Build the Docker image**

Before building the image, you may modify the [Dockerfile](Dockerfile) according to your CUDA version.
The CUDA version we use is 11.6. You can change the base image tag according to [this site](https://hub.docker.com/r/nvidia/cuda/tags).
You also need to change the `--extra-index-url` according to your CUDA version in order to install the correct version of Pytorch.
You can check the url through [Pytorch website](https://pytorch.org).

Use this command to build the image

```shell
cd Semi-supervised-learning && docker build -t semilearn .
```

Job done. You can use the image you just built for your own project. Don't forget to use the argument `--gpu` when you want
to use GPU in a container.

### Training

Here is an example to train ILF-HCTN on MOF-Cohort with Survival Analysis label:

```sh
#Train F-ViT
python train.py --c exp1/nsclccox-fusion-finetune.yaml
#Train A-ViT
python train.py --c exp1/nsclccox-fusion-mona.yaml
#Train ILF-HCTN, "pretrain_path_finetune" in yaml file should be set to the real path.
python train.py --c exp1/nsclccox-fusion-dual.yaml
```

### Evaluation

After training, you can check the evaluation performance on training logs, or running evaluation script:

```
python evaluate.py --c exp0/semi-fusioncox-dual.yaml
```

<!-- MODEL ZOO -->

## Model Zoo

TODO: add pre-trained models.

<p align="right">(<a href="#top">back to top</a>)</p>

# Framework: USB

<!-- PROJECT LOGO -->

<br />

<div align="center">
  <a href="https://github.com/microsoft/Semi-supervised-learning">
    <img src="figures/logo.png" alt="Logo" width="400">
  </a>


<!-- <h3 align="center">USB</h3> -->

<p align="center">
    <strong>USB</strong>: A Unified Semi-supervised learning Benchmark for CV, NLP, and Audio Classification
    <!-- <br />
    <a href="https://github.com/microsoft/Semi-supervised-learning"><strong>Explore the docs »</strong></a>
    <br /> -->
    <br />
    <a href="https://arxiv.org/abs/2208.07204">Paper</a>
    ·
    <a href="https://github.com/microsoft/Semi-supervised-learning/tree/main/results">Benchmark</a>
    ·
    <a href="https://colab.research.google.com/drive/1lFygK31jWyTH88ktao6Ow-5nny5-B7v5">Demo</a>
    ·
    <a href="https://usb.readthedocs.io/en/main/">Docs</a>
    ·
    <a href="https://github.com/microsoft/Semi-supervised-learning/issues">Issue</a>
    ·
    <a href="https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/articles/pushing-the-limit-of-semi-supervised-learning-with-the-unified-semi-supervised-learning-benchmark/">Blog</a>
    ·
    <a href="https://medium.com/p/849f42bbc32a">Blog (Pytorch)</a>
    ·
    <a href="https://zhuanlan.zhihu.com/p/566055279">Blog (Chinese)</a>
    ·
    <a href="https://nips.cc/virtual/2022/poster/55710">Video</a>
    ·
    <a href="https://www.bilibili.com/video/av474982872/">Video (Chinese)</a>
  </p>

</div>

<!-- Introduction -->

## Introduction

**USB** is a Pytorch-based Python package for Semi-Supervised Learning (SSL). It is easy-to-use/extend, *affordable* to small groups, and comprehensive for developing and evaluating SSL algorithms. USB provides the implementation of 14 SSL algorithms based on Consistency Regularization, and 15 tasks for evaluation from CV, NLP, and Audio domain.

![Code Structure](./figures/code.png)


<p align="right">(<a href="#top">back to top</a>)</p>

### Develop

Check the developing documentation for creating your own SSL algorithm!

_For more examples, please refer to the [Documentation](https://example.com)_

<!-- TRADEMARKS -->

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft
trademarks or logos is subject to and must follow
[Microsoft&#39;s Trademark &amp; Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.

<!-- LICENSE -->

## License

Distributed under the MIT License. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- ACKNOWLEDGMENTS -->

## Acknowledgments

We thanks the following projects for reference of creating USB:

- [TorchSSL](https://github.com/TorchSSL/TorchSSL)
- [FixMatch](https://github.com/google-research/fixmatch)
- [CoMatch](https://github.com/salesforce/CoMatch)
- [SimMatch](https://github.com/KyleZheng1997/simmatch)
- [HuggingFace](https://huggingface.co/docs/transformers/index)
- [Pytorch Lighting](https://github.com/Lightning-AI/lightning)
- [README Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#top">back to top</a>)</p>

<!-- MARKDOWN LINKS & IMAGES -->

<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[contributors-url]: https://github.com/microsoft/Semi-supervised-learning/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[forks-url]: https://github.com/microsoft/Semi-supervised-learning/network/members
[stars-shield]: https://img.shields.io/github/stars/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[stars-url]: https://github.com/microsoft/Semi-supervised-learning/stargazers
[issues-shield]: https://img.shields.io/github/issues/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[issues-url]: https://github.com/microsoft/Semi-supervised-learning/issues
[license-shield]: https://img.shields.io/github/license/microsoft/Semi-supervised-learning.svg?style=for-the-badge
[license-url]: https://github.com/microsoft/Semi-supervised-learning/blob/main/LICENSE.txt
