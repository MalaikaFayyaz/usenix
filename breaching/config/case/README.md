# Our Evaluated Cases

Here are the configurations we have evaluated with in our experiments. 

Please refers to the showcase in our [root directory](./../../../README.md) for usages, and [JonasGeiping/breaching](https://github.com/JonasGeiping/breaching) for further customizations.



### 🎯 1. Initialization Variations

**CelebA-HQ**

- *uniform*: [celeba_hq_identity_random_untrained_kaiming_uniform_resnet18.yaml](./celeba_hq_identity_random_untrained_kaiming_uniform_resnet18.yaml)
- *normal*: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18.yaml)
- *orthogonal*: [celeba_hq_identity_random_untrained_orthogonal_resnet18.yaml](./celeba_hq_identity_random_untrained_orthogonal_resnet18.yaml)
- *transferred*: [celeba_hq_identity_random_untrained_transferred_resnet18.yaml](./celeba_hq_identity_random_untrained_transferred_resnet18.yaml)



**CIFAR-10**

- *uniform*: [cifar10_random_untrained_kaiming_uniform_resnet18.yaml](./cifar10_random_untrained_kaiming_uniform_resnet18.yaml)
- *normal*: [cifar10_random_untrained_kaiming_normal_resnet18.yaml](./cifar10_random_untrained_kaiming_normal_resnet18.yaml)
- *orthogonal*: [cifar10_random_untrained_orthogonal_resnet18.yaml](./cifar10_random_untrained_orthogonal_resnet18.yaml)



**Other datasets (Normal Initializations)**

- **LC25000**: [lc25000_random_untrained_kaiming_normal_resnet18.yaml](./lc25000_random_untrained_kaiming_normal_resnet18.yaml)
- **ImageNet**: [imagenet_random_untrained_kaiming_normal_resnet18.yaml](./imagenet_random_untrained_kaiming_normal_resnet18.yaml)
- **CIFAR-100**: [cifar100_random_untrained_kaiming_normal_resnet18.yaml](./cifar100_random_untrained_kaiming_normal_resnet18.yaml)



### 🎯 2. FL’s Lifecycle

Please refer to [utils/model](./../../../utils/model/README.md) for detailed instructions. 



### 🎯 3. Varying Batch Sizes (Step $=1$)

**CelebA-HQ**

- Batch Size $= 1$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18.yaml)
- Batch Size $= 2$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_2_epoch_1.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_2_epoch_1.yaml)
- Batch Size $= 4$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_4_epoch_1.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_4_epoch_1.yaml)
- Batch Size $= 8$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_8_epoch_1.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_8_epoch_1.yaml)
- Batch Size $= 16$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_16_epoch_1.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_16_epoch_1.yaml)
- Batch Size $= 32$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_32_epoch_1.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_32_epoch_1.yaml)



**CIFAR-10**

- Batch Size $= 1$: [cifar10_random_untrained_kaiming_normal_resnet18.yaml](./cifar10_random_untrained_kaiming_normal_resnet18.yaml)
- Batch Size $= 2$: [cifar10_random_untrained_kaiming_normal_resnet18_batch_2_epoch_1.yaml](./cifar10_random_untrained_kaiming_normal_resnet18_batch_2_epoch_1.yaml)
- Batch Size $= 4$: [cifar10_random_untrained_kaiming_normal_resnet18_batch_4_epoch_1.yaml](./cifar10_random_untrained_kaiming_normal_resnet18_batch_4_epoch_1.yaml)
- Batch Size $= 8$: [cifar10_random_untrained_kaiming_normal_resnet18_batch_8_epoch_1.yaml](./cifar10_random_untrained_kaiming_normal_resnet18_batch_8_epoch_1.yaml)
- Batch Size $= 16$: [cifar10_random_untrained_kaiming_normal_resnet18_batch_16_epoch_1.yaml](./cifar10_random_untrained_kaiming_normal_resnet18_batch_16_epoch_1.yaml)
- Batch Size $= 32$: [cifar10_random_untrained_kaiming_normal_resnet18_batch_32_epoch_1.yaml](./cifar10_random_untrained_kaiming_normal_resnet18_batch_32_epoch_1.yaml)



### 🎯 4. Varying Batch Sizes and Steps

**CelebA-HQ**

- Batch Size $= 1$, Step $=2$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_1_epoch_2.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_1_epoch_2.yaml)
- Batch Size $= 1$, Step $=3$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_1_epoch_3.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_1_epoch_3.yaml)
- Batch Size $= 2$, Step $=2$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_2_epoch_2.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_2_epoch_2.yaml)
- Batch Size $= 2$, Step $=3$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_2_epoch_3.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_2_epoch_3.yaml)
- Batch Size $= 4$, Step $=2$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_4_epoch_2.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_4_epoch_2.yaml)
- Batch Size $= 4$, Step $=3$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_4_epoch_3.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_4_epoch_3.yaml)
- Batch Size $= 8$, Step $=2$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_8_epoch_2.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_8_epoch_2.yaml)
- Batch Size $= 8$, Step $=3$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_8_epoch_3.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_batch_8_epoch_3.yaml)



### 🎯 5. Models Generality

- **ResNet-18**: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18.yaml)
- **ResNet-34**: [celeba_hq_identity_random_untrained_kaiming_normal_resnet34.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet34.yaml)
- **DesNet-121**: [celeba_hq_identity_random_untrained_kaiming_normal_desnet121.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_desnet121.yaml)
- **MobileNet-v2**: [celeba_hq_identity_random_untrained_kaiming_normal_mobilenet_v2.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_mobilenet_v2.yaml)



### 🎯 6. Defenses Evaluations (Pruning)

**Magnitude**

- $p=10\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.1.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.1.yaml)
- $p=30\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.3.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.3.yaml)
- $p=50\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.5.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.5.yaml)
- $p=70\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.7.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.7.yaml)
- $p=90\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.9.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_magnitude_0.9.yaml)



**Soteria**

- $p=10\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.1.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.1.yaml)
- $p=30\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.3.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.3.yaml)
- $p=50\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.5.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.5.yaml)
- $p=70\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.7.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.7.yaml)
- $p=90\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.9.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_soteria_0.9.yaml)



**OUTPOST**

- $p=10\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.1.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.1.yaml)
- $p=30\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.3.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.3.yaml)
- $p=50\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.5.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.5.yaml)
- $p=70\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.7.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.7.yaml)
- $p=90\\%$: [celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.9.yaml](./celeba_hq_identity_random_untrained_kaiming_normal_resnet18_pruning_outpost_0.9.yaml)

