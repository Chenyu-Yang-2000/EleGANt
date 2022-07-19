# Preparation Instructions

Clone this repository and prepare the dataset and weights through the following steps:

**a. Prepare model weights for face detection.**

Download the weights of [dlib](https://github.com/davisking/dlib) face detector of 68 landmarks [here](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2). Unzip it and move it to the directory `./faceutils/dlibutils`.

Download the weights of BiSeNet ([PyTorch implementation](https://github.com/zllrunning/face-parsing.PyTorch)) for face parsing [here](https://drive.google.com/open?id=154JgKpzCPW82qINcVieuPH3fZ2e0P812). Rename it as `resnet.pth` and move it to the directory `./faceutils/mask`.

**b. Prepare Makeup Transfer (MT) dataset.**

Download raw data of the MT Dataset [here](https://github.com/wtjiang98/PSGAN) and unzip it into sub directory `./data`.

Run the following command to preprocess data:

```bash
python training/preprocess.py
```

Your data directory should look like:

```text
data
└── MT-Dataset
    ├── images
    │   ├── makeup
    │   └── non-makeup
    ├── segs
    │   ├── makeup
    │   └── non-makeup
    ├── lms
    │   ├── makeup
    │   └── non-makeup
    ├── makeup.txt
    ├── non-makeup.txt
    └── ...
```

**c. Download weights of trained EleGANt.**

The weights of our trained model can be download [here](https://drive.google.com/drive/folders/1xzIS3Dfmsssxkk9OhhAS4svrZSPfQYRe?usp=sharing). Put it under the directory `./ckpts`.
