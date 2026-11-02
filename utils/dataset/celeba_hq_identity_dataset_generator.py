"""
@Encoding:      UTF-8 
@File:          celeba_hq_identity_dataset_generator.py

@Introduction:  Transpose CelebA-HQ to CelebA-HQ-Identity Dataset.
@Author:        Kaiming Zhu
@Reference:     [1] https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/blob/main/identity_dataset_generator.py
                [2] https://github.com/ndb796/CelebA-HQ-Face-Identity-and-Attributes-Recognition-PyTorch/blob/main/CelebAMask_HQ_Dataset_Processing.ipynb
@Notes:         1. Please Install CelebA-HQ Dataset by instructions in [2] before you run this scripts.
                2. run `unzip CelebAMask-HQ.zip` before you run it.
"""

import os
from shutil import copyfile
from pathlib import Path


# 1. Where to load the source dataset (CelebAMask-HQ).
identity_file_path = f"{str(Path.home())}/data/celeba-hq/CelebA-HQ-identity.txt"
source_dataset_path = f"{str(Path.home())}/data/celeba-hq/CelebAMask-HQ/CelebA-HQ-img/"

# 2. Where to save the target dataset (CelebA-HQ-Identity).
identity_dataset_save_path = f"{str(Path.home())}/data/celeba-hq-identity/"

# 3. Configs on dataset generating
train_ratio = 0.8       # train-test split (ratio of training set).
least_image = 15        # filter identities, whose amount of datapoints are less than it.


if __name__ == "__main__":
    # Step 1. Load identities
    identities = {}
    with open(identity_file_path) as f:
        lines = f.readlines()
        for line in lines:
            file_name, identity = line.strip().split()
            identities[file_name] = identity

    print("Finished Identity Checking")
    print(f'There are {len(set(identities.values()))} identities.')
    print(f'There are {len(identities.keys())} images.')

    # Step 2. Make result folders, and copy files from the source dataset.
    file_list = os.listdir(source_dataset_path)
    for file in file_list:
        identity = identities[file]
        source = os.path.join(source_dataset_path, file)
        target = os.path.join(identity_dataset_save_path, str(identity), file)
        if not os.path.exists(os.path.join(identity_dataset_save_path, str(identity))):
            os.makedirs(os.path.join(identity_dataset_save_path, str(identity)))
        copyfile(source, target)

    # Step 3. Generate dataset and make train-test split
    folder_list = os.listdir(identity_dataset_save_path)
    identity_count = 0
    train_images = 0
    test_images = 0
    for folder in folder_list:
        file_list = os.path.join(identity_dataset_save_path, folder)
        file_list = os.listdir(file_list)
        if len(file_list) < least_image:
            continue
            
        identity_count += 1
        num_train = int(train_ratio * len(file_list))
        for file in file_list[:num_train]:
            train_images += 1
            source = os.path.join(identity_dataset_save_path, folder, file)
            target = os.path.join(identity_dataset_save_path, 'train', folder, file)
            if not os.path.exists(os.path.join(identity_dataset_save_path, 'train', folder)):
                os.makedirs(os.path.join(identity_dataset_save_path, 'train', folder))
            os.rename(source, target)
        for file in file_list[num_train:]:
            test_images += 1
            source = os.path.join(identity_dataset_save_path, folder, file)
            target = os.path.join(identity_dataset_save_path, 'test', folder, file)
            if not os.path.exists(os.path.join(identity_dataset_save_path, 'test', folder)):
                os.makedirs(os.path.join(identity_dataset_save_path, 'test', folder))
            os.rename(source, target)

    print("\nFinished dataset splitting")
    print(f'There are {identity_count} identities that have more than {least_image} images.')
    print(f'There are {train_images} train images.')
    print(f'There are {test_images} test images.')