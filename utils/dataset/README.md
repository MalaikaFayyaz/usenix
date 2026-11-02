# [🛠️Optional] Install Datasets
- ✅ If you just need to experiment with **CIFAR-10** and **CIFAR-100** dataset, you could skip this instruction.
- 🛠️ If you need to experiment with **ImageNet-ILSVRC2012**, **CelebA-HQ** and **LC25000** dataset, please follows the instruction below. All guidances in this instruction would save your installations to `~/data` by default (and `~/data` would be automatically created if not existed).

### Step 1. Installations

**ImageNet-ILSVRC2012**

Please refers to [https://image-net.org/download.php](https://image-net.org/download.php).


**CelebA-HQ**

> *Update: Since [version v2](https://zenodo.org/records/17979356), we have released the file `[extension]CelebA-HQ.zip`. You can directly reuse this released zipped dataset by following the instructions included in it.*
>
> *The instructions below will also guide you to generate it from the source files.*

1. Execute following commands in bash to download *CelebA-HQ* .
2. Run `./celeba_hq_identity_dataset_generator.py`, segmenting it by the identity.

```bash
mkdir ~/data/celeba-hq
cd ~/data/celeba-hq

# Snapshots from: https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/blob/main/CelebAMask_HQ_Dataset_Processing.ipynb
wget https://postechackr-my.sharepoint.com/:u:/g/personal/dongbinna_postech_ac_kr/Eb37jNPPA7hHl0fmktYqcV8B-qmPLx-ZKYQ1eFk4UPBV_A?download=1 -O CelebAMask-HQ.zip
wget https://postechackr-my.sharepoint.com/:t:/g/personal/dongbinna_postech_ac_kr/EVRoUY8_txRFv56-KWvZrksBDWbD6adkjBxwwRN7qAC6bg?download=1 -O CelebA-HQ-identity.txt
wget https://postechackr-my.sharepoint.com/:t:/g/personal/dongbinna_postech_ac_kr/EVrdIrPOkR1OlEWBVK8lE3AB9bFh741GnKBkNgPa8trNuA?download=1 -O CelebA-HQ-attribute.txt

unzip CelebAMask-HQ.zip
```

**Lung and Colon Cancer (LC25000)**

Execute following commands in bash to download *LC25000* .

```bash
# The script would download it to `~/data/lung_colon_image_set` by default.
# Please checkout `./lc25000_downloader.py` to change to an alternative path.
python lc25000_downloader.py
```

### Step 2. Linking to configs in `breaching`

Please locate to following configs, set the value of `path` to where this dataset locates. An example is provided below.

- **ImageNet-ILSVRC2012** : `./breaching/config/case/data/ImageNet.yaml`
- **CelebA-HQ** : `./breaching/config/case/data/CelebA-HQ-Identity.yaml`
- **LC25000** : `./breaching/config/case/data/LC25000.yaml`

```yaml
#
# @ Notes: Example to set path of **CelebA-HQ** dataset
# @ Path: ./breaching/config/case/data/CelebA-HQ-Identity.yaml
#

name: CelebA-HQ-Identity
modality: vision
task: classification

# [🌟Here!] Change this key
path: "path it located" # e.g. "~/data/celeba-hq-identity"

```