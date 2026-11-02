# Cracking Federated Privacy: Initialization-Resilient Gradient Inversion with Fine-Grained Reconstruction
- ℹ️ **Introduction:** This is the official implementation of paper "Cracking Federated Privacy: Initialization-Resilient Gradient Inversion with Fine-Grained Reconstruction" (USENIX Sec.'26).

- 🖥️ **Environments**: Our experiments are conducted by Python 3.9.18, PyTorch 1.13.1, and CUDA 11.7.

## 🚩 How to start

#### 1. 🌟 Create Python Environment (Anaconda)
```bash
# 1. Open bash, and locate it to the root path of this directory.
cd ~/CFGIA

# 2. Run following command.
conda env create -f environment.yaml
```

#### 2. 🛠️ Dataset Installations
We have evaluated on $5$ different dataset.

- ✅ **CIFAR-10** and **CIFAR-100**. These dataset would automatically installed & reused whenever you run with related configurations.

- 🛠️ **ImageNet-ILSVRC2012**, **CelebA-HQ** and **LC25000**. Please refers to the [README.md](./utils/dataset/README.md) file in `./utils/dataset/`, where we have provided the guidance to install them.


#### 3. 🎯 Attack (Reproducing)

- To facilitate your evaluation, we have provided several kinds of showcase in the root directory. They are aligned with the experiment result in the *Fig. 5*, which contains:
    - [Ours](<./[Fig.5] Ours.ipynb>)
    - [HFGradInv(AAAI'24)](<./[Fig.5] HFGradInv(AAAI'24).ipynb>)
    - [SAPAG(IJCNN'22)](<./[Fig.5] SAPAG(IJCNN'22).ipynb>)
    - [iDLG(Arxiv'20)](<./[Fig.5] iDLG(Arxiv'20).ipynb>)
    - [InvG(NeurIPS'20)](<./[Fig.5] InvG(NeurIPS'20).ipynb>)

- To perform evaluations as reported in other figures, please refers to [./breaching/config/case](./breaching/config/case/README.md), it contains most cases we have been evaluated with.

## ❤️ Acknowledgement

- This repo is an extension from [JonasGeiping/breaching](https://github.com/JonasGeiping/breaching).

- Thanks the authors of [*AGIC: Approximate Gradient Inversion Attack on Federated Learning*](
https://ieeexplore.ieee.org/document/9996844) for kindly sharing their implementation, which facilitated our experimental evaluation.


## ✍️ How to cite us

We appreciate your interest in our work. 

It is our pleasure to motivate your further research❤️.

```bibtex
@inproceedings{zhu2026cracking,
  title={Cracking Federated Privacy: Initialization-Resilient Gradient Inversion with Fine-Grained Reconstruction},
  author={Kaiming, Zhu and Jinsheng, Yang and Siyang, Guo and Huaqian, Qin and Taiyu, Wang and Junbo, Wang and Yuhong, Nan and Zibin, Zheng},
  booktitle={the 35th USENIX Security Symposium},
  pages={},
  year={2026},
  organization={USENIX}
}
```







