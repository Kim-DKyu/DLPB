import math
import os
import random
from math import floor

import numpy
import torch
import torch.nn.functional as F
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import default_collate, DataLoader

from torchvision import transforms
from torchvision.datasets import ImageFolder, CIFAR10, CIFAR100, FashionMNIST
from torchvision.datasets.samplers import DistributedSampler
from torchvision.transforms import RandomChoice

_dataset_dict = {
    'ImageFolder': ImageFolder,
    'CIFAR10': CIFAR10,
    'CIFAR100': CIFAR100,
    'FashionMNIST': FashionMNIST,
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


class TrainTransform:
    def __init__(self, resize, resize_mode, pad, scale, ratio, hflip, auto_aug, remode, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        transform_list = []

        if hflip:
            transform_list.append(transforms.RandomHorizontalFlip(hflip))

        if auto_aug:
            if auto_aug.startswith('ra'):
                transform_list.append(transforms.RandAugment(interpolation=interpolation))
            elif auto_aug.startswith('ta_wide'):
                transform_list.append(transforms.TrivialAugmentWide(interpolation=interpolation))
            elif auto_aug.startswith('aa'):
                policy = transforms.AutoAugmentPolicy('imagenet')
                transform_list.append(transforms.AutoAugment(policy=policy, interpolation=interpolation))

        if resize_mode == 'RandomResizedCrop':
            transform_list.append(transforms.RandomResizedCrop(resize, scale=scale, ratio=ratio, interpolation=interpolation))
        elif resize_mode == 'ResizeRandomCrop':
            transform_list.extend([transforms.Resize(resize, interpolation=interpolation),
                                   transforms.RandomCrop(resize, padding=pad)])
        elif resize_mode == 'RandomCrop':
            transform_list.append(transforms.RandomCrop(resize, padding=pad))
        else:
            assert f"{resize_mode} should be RandomResizedCrop and ResizeRandomCrop"

        transform_list.extend([
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])

        if remode:
            transform_list.append(transforms.RandomErasing(remode))

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class ValTransform:
    def __init__(self, size, resize_mode, crop_ptr, interpolation, mean, std):
        interpolation = transforms.functional.InterpolationMode(interpolation)

        if not isinstance(size, (tuple, list)):
            size = (size, size)

        resize = (int(floor(size[0] / crop_ptr)), int(floor(size[1] / crop_ptr)))

        if resize_mode == 'resize_shorter':
            resize = resize[0]

        transform_list = [
                transforms.ToTensor(),
                transforms.Normalize(mean, std)
            ]

        self.transform_fn = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transform_fn(x)


class MixUP:
    def __init__(self, p=0.5, alpha=1.0, nclass=1000):
        self.p = p
        self.alpha = alpha
        self.nclass = nclass

    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.nclass).to(dtype=batch.dtype)

        ratio = float(1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        batch = batch * (1-ratio) + batch_roll * ratio
        target = target * (1-ratio) + target_roll * ratio

        return batch, target


class CutMix:
    def __init__(self, p=0.5, alpha=1.0, nclass=1000):
        self.p = p
        self.alpha = alpha
        self.nclass = nclass

    @torch.inference_mode()
    def __call__(self, batch, target):
        if self.p > random.random():
            return batch, target

        if target.ndim == 1:
            target = F.one_hot(target, num_classes=self.nclass).to(dtype=batch.dtype)

        B, C, H, W = batch.shape
        ratio = float(1 - torch._sample_dirichlet(torch.tensor([self.alpha, self.alpha]))[0])

        batch_roll = batch.roll(1, 0)
        target_roll = target.roll(1, 0)

        height_half = int(0.5 * math.sqrt(ratio) * H)
        width_half = int(0.5 * math.sqrt(ratio) * W)
        r = int(random.random() * H)
        c = int(random.random() * W)

        start_x = max(r - height_half, 0)
        end_x = min(r + height_half, H)
        start_y = max(r - width_half, 0)
        end_y = min(r + width_half, W)

        ratio = 1 - ((end_x - start_x) * (end_y - start_y) / (H * W))

        batch[:, :, start_x:end_x, start_y:end_y] = batch_roll[:, :, start_x:end_x, start_y:end_y]
        target = target * (1-ratio) + target_roll * ratio

        return batch, target

def get_dataloader(train_dataset, val_dataset, args):
    seed = args.seed
    rank = getattr(args, 'rank', 0)

    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(seed)
    train_worker_generator = torch.Generator()
    train_worker_generator.manual_seed(seed + rank)
    val_worker_generator = torch.Generator()
    val_worker_generator.manual_seed(seed + 10_000 + rank)

    # 1. create sampler
    if args.distributed:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=seed)
        val_sampler = DistributedSampler(val_dataset, shuffle=False, seed=seed)
    else:
        train_sampler = RandomSampler(train_dataset, generator=sampler_generator)
        val_sampler = SequentialSampler(val_dataset)

    # 2. create collate_fn
    mix_collate = []
    if args.mixup:
        mix_collate.append(MixUP(alpha=args.mixup, nclass=args.num_classes))
    if args.cutmix:
        mix_collate.append(CutMix(alpha=args.mixup, nclass=args.num_classes))

    if mix_collate:
        mix_collate = RandomChoice(mix_collate)
        collate_fn = lambda batch: mix_collate(*default_collate(batch))
    else:
        collate_fn = None

    # 3. create dataloader
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=False, sampler=train_sampler,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=args.pin_memory,
        worker_init_fn=seed_worker, generator=train_worker_generator)

    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False, sampler=val_sampler,
        num_workers=args.num_workers, collate_fn=None, pin_memory=False,
        worker_init_fn=seed_worker, generator=val_worker_generator)

    args.iter_per_epoch = len(train_dataloader)

    return train_dataloader, val_dataloader


def get_dataset(args):
    dataset_class = _dataset_dict[args.dataset_type]
    train_transform = TrainTransform(args.train_size, args.train_resize_mode, args.random_crop_pad, args.random_crop_scale, args.random_crop_ratio, args.hflip, args.auto_aug, args.remode, args.interpolation, args.mean, args.std)
    val_transform = ValTransform(args.test_size, args.test_resize_mode, args.center_crop_ptr, args.interpolation, args.mean, args.std)
    if args.dataset_type == 'ImageFolder':
        train_dataset = dataset_class(os.path.join(args.data_dir, args.train_split), train_transform)
        val_dataset = dataset_class(os.path.join(args.data_dir, args.val_split), val_transform)
        args.num_classes = len(train_dataset.classes)
    elif args.dataset_type in _dataset_dict.keys():
        train_dataset = dataset_class(root=args.data_dir, train=True, download=True, transform=train_transform)
        val_dataset = dataset_class(root=args.data_dir, train=False, download=True, transform=val_transform)
        args.num_classes = len(train_dataset.classes)
    else:
        assert f"{args.dataset_type} is not supported yet. Just make your own code for it"

    return train_dataset, val_dataset
