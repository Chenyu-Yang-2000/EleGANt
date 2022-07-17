# EleGANt: Exquisite and Locally Editable GAN for Makeup Transfer

Official [PyTorch](https://pytorch.org/) implementation of ECCV 2022 paper: [**EleGANt: Exquisite and Locally Editable GAN for Makeup Transfer**]

*Chenyu Yang, Wanrong He, Yingqing Xu, and Yang Gao*.

![teaser](assets/figs/teaser.png)

## Getting Started

- [Installation](assets/docs/install.md)
- [Prepare Dataset & Checkpoints](assets/docs/prepare.md)

## Test

To test our model, download the [weights](https://drive.google.com/drive/folders/1xzIS3Dfmsssxkk9OhhAS4svrZSPfQYRe?usp=sharing) of the trained model and run

```bash
python scripts/demo.py
```

## Train

To train a model from scratch, run

```bash
python scripts/train.py
```

## More Results

**Controllable makeup transfer.**

![control](assets/figs/control.png 'controllable makeup transfer')

**Local makeup editing.**

![edit](assets/figs/edit.png 'local makeup editing')

## Citation

Coming soon.

## Acknowledgement

Some of the codes are build upon [PSGAN](https://github.com/wtjiang98/PSGAN) and [aster.Pytorch](https://github.com/ayumiymk/aster.pytorch).
