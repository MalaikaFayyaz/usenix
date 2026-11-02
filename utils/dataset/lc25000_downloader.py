"""
@Encoding:      UTF-8 
@File:          lc25000_downloader.py

@Introduction:  Simple script help downloading "Lung and colon histopathological image dataset" (LC25000).
@Author:        Kaiming Zhu
@Date:          2025/12/8 02:37
@Reference:     [1] Where to download: https://www.kaggle.com/datasets/javaidahmadwani/lc25000/data
                [2] How to train on it (optional): https://www.kaggle.com/code/pooyaebrahimi/resnet-cancer-detection.
@Notes:         Its size is `1.76`GB.
"""

import os
from pathlib import Path
import shutil

import kagglehub

path_to_achrive = str(Path.home()) + "/data" # Should not End with "/"

# Download latest version
download_path = kagglehub.dataset_download("javaidahmadwani/lc25000")

# Copy downloaded files to `path_to_archive`
shutil.copytree(download_path, path_to_achrive, dirs_exist_ok=True)
shutil.rmtree(download_path)

