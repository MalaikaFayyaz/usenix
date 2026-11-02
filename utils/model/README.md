# [🔬Optional] Life-cycle Evaluations

We evaluate attack performance across the FL life cycle by generating models with different test accuracies. Each model is used as a proxy for a particular stage of FL training.

As the global model typically becomes more accurate over aggregation rounds, this setup helps illustrate how attack effectiveness changes as training progresses.

Notes: You could skip this file if you just need to experiment with initializations variations.

### 🎯 Step 1. Install Datasets

For **CelebA-HQ-Identity**, please refer to [README](./../dataset/README.md) in dataset for installations.

### 🎯 Step 2. Train & Save Models

> *Update: Since [version v2](https://zenodo.org/records/17979356), we have released the file `[extension]our_evaluated_models.zip`. You can directly reuse this released zipped models by following the instructions included in it.*
>
> *The instructions below will also guide you to generate it from the source files.*

All models would be saved under this folder, named as:

- **CIFAR10:** `./cifar10/`
- **CelebA-HQ-Identity:** `./celeba_hq_{res}x{res}/`(res: resolution of it, $224$ by default).

For how to train models:

- **CIFAR10:** Please refer to [./cifar10_resnet18_trainer.py](./cifar10_resnet18_trainer.py).

- **CelebA-HQ-Identity:** Please refer to [./celeba_identity_resnet18_trainer.py](./celeba_identity_resnet18_trainer.py).



### 🎯 Step 3. How to evaluate on the model you trained?

**Short Answer:** 

- Create a new case config in [/CFGIA/breaching/config/case/](../../breaching/config/case/).
- Specify the path to the model you trained in this new config, by setting `state_dict_path`. 
- Use this new config to evaluate. Please refer to our provided showcases in the [README](./../../README.md) of repo.

Here are the configurations to specify a model in your evaluation (Batch Size $=1$, Local Step $=1$), which are served as templates.

**CIFAR10**

```yaml
# [🌟Change this key!] Give it a custom name.
# @Examples:
# - test
# - cifar10_pretrained
name: your_specified_name

defaults:
  - data: CIFAR10_trained_random
  - impl: default
  - server: honest-but-curious_transferred
  - user: local_gradient
  - _self_

model: ResNet18

# [🌟Change this key!] absolute path to model.
# @Examples:
# - "/home/user1/CFGIA/util/model/cifar10/epoch_0.pt"
# - "/home/user2/dev/CFGIA/util/model/cifar10/best.pt"
#
# @Notes: Consider using the command `pwd` to look for its path in Linux.
state_dict_path: absolute_path_to_model

# Server and user:
num_queries: 1
```

**CelebA-HQ-Identity**

```yaml
# [🌟Change this key!] Give it a custom name.
# @Examples:
# - test
# - celeba_hq_identity_pretrained
name: your_specified_name

defaults:
  - data: celeba_hq_identity_random
  - impl: default
  - server: honest-but-curious_transferred
  - user: local_gradient
  - _self_

model: ResNet18

# [🌟Change this key!] absolute path to model.
# @Examples:
# - "/home/user1/CFGIA/util/model/celeba_hq_224x224/epoch_0.pt"
# - "/home/user2/dev/CFGIA/util/model/celeba_hq_224x224/best.pt"
#
# @Notes: 
# - Consider using the command `pwd` to look for its path in Linux.
state_dict_path: "absolute_path_to_model"

# Server and user:
num_queries: 1
```

**What's more** 

You could also use the hyper-link (i.e. "https://xxx" or "http://xxx") to specify the path of a released model (if they have the same structure). 

For this template, please refer to [cifar10_random_trained_resnet18.yaml](./../../breaching/config/case/cifar10_random_trained_resnet18.yaml), where we have provided the configuration to evaluate with a released CIFAR-10 model (test accuracy $=95\%$).
