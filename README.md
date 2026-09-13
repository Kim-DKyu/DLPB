# Knowledge Distillation with Lightweight Peer Branches for Learning Diverse Representations

Official implementation of **Knowledge Distillation with Lightweight Peer Branches for Learning Diverse Representations**. DLPB trains multiple lightweight peer branches inside a shared backbone and uses their complementary predictions for self-distillation. After training, the peer branches can be removed so inference uses the backbone model.

## Requirements

The code was developed and tested with Python 3.10, PyTorch 1.13.1, and CUDA 11.6.

```bash
pip install -r requirements.txt
```

Training requires an NVIDIA GPU with CUDA support.

## Data

The training presets use CIFAR-100. The default commands expect the data directory `data`. The dataset loader downloads the dataset when it is not available locally.

## Code Structure

```text
DLPB/
├── args.py, dataset.py, optimizer.py
├── metric.py, utils.py, setup.py, log.py
├── models/
│   ├── resnet32_dlpb.py
│   ├── resnet110_dlpb.py
│   ├── wideresnet_dlpb.py
│   └── densenet_dlpb.py
├── train/
│   ├── train_resnet32_dlpb.py
│   ├── train_resnet110_dlpb.py
│   ├── train_wideresnet_dlpb.py
│   └── train_densenet_dlpb.py
└── multi_train/
    ├── multi_train_resnet32_dlpb.py
    ├── multi_train_resnet110_dlpb.py
    ├── multi_train_wideresnet_dlpb.py
    └── multi_train_densenet_dlpb.py
```

The root files provide shared data loading, optimization, metrics, distributed setup, logging, and checkpoint utilities. Model definitions are in `models/`. Single-run and multi-run training entry points are in `train/` and `multi_train/`.

## Training

### ResNet-32

```bash
torchrun --nproc_per_node=2 multi_train/multi_train_resnet32_dlpb.py \
  cifar100_resnet32_dlpb \
  -m ga_resnet32_gram_dim_32_branch_3 \
```

### ResNet-110

```bash
torchrun --nproc_per_node=2 multi_train/multi_train_resnet110_dlpb.py \
  cifar100_resnet110_dlpb \
  -m ga_resnet110_gram_dim_64_branch_3 \
```

### WideResNet

```bash
torchrun --nproc_per_node=2 multi_train/multi_train_wideresnet_dlpb.py \
  cifar100_wideresnet_dlpb \
  -m wide_resnet20_8_ga_64_branch_3 \
```

### DenseNet-40-k12

```bash
torchrun --nproc_per_node=2 multi_train/multi_train_densenet_dlpb.py \
  cifar100_densenet40k12_dlpb \
  -m ga_densenetd40k12_branch_3 \
```

The default presets use CIFAR-100, 300 epochs, SGD with Nesterov momentum, a multistep learning-rate schedule, and three peer branches. Use `python multi_train/<entry-point>.py --help` to inspect available options.

## Evaluation

Training outputs and checkpoints are written to the directory specified by `-o`. Validation and test metrics are reported during execution. Checkpoint evaluation uses the options supported by the corresponding training entry point.

## Results

The main paper evaluates DLPB with ResNet-32, ResNet-110, WideResNet, and DenseNet backbones on CIFAR-100. The final result table will be added with the paper release.

## License

This project is released under the Apache License 2.0. See [LICENSE](LICENSE) for details.

## Code Availability

The source code is publicly available at [https://github.com/Kim-DKyu/DLPB](https://github.com/Kim-DKyu/DLPB).
