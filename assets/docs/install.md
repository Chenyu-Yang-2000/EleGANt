# Installation Instructions

This code was tested on Ubuntu 20.04 with CUDA 11.1.

**a. Create a conda virtual environment and activate it.**

```bash
conda create -n elegant python=3.8
conda activate elegant
```

**b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**

```bash
pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```

**c. Install other required libaries.**

```bash
pip install opencv-python matplotlib dlib fvcore
```
