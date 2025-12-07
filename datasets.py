"""
IEOR4540: 
    This is based on code from https://github.com/naver-ai/rope-vit/tree/main/deit

    I added logic specific to CIFAR10
"""

import os 

from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD


def build_dataset(is_train, args):
    transform = build_transform(is_train, args)

    if args.data_set == "CIFAR10":
        dataset = datasets.CIFAR10(root = "data/", 
                                    train=is_train, 
                                    transform=transform,
                                    download = True)
        nb_classes = 10
    elif args.data_set == "CIFAR100":
        dataset = datasets.CIFAR100(root = "data/", 
                                    train=is_train, 
                                    transform=transform,
                                    download = True)
        nb_classes = 100
    elif args.data_set == "MNIST":
        dataset = datasets.MNIST(root = "data/", 
                                    train=is_train, 
                                    transform=transform,
                                    download = True)
        nb_classes = 10
    else:
        raise Exception("Invalid dataset.")
    
    return dataset, nb_classes


def build_transform(is_train, args):
    resize_im = args.input_size > 32
    
    # MNIST SPECIFIC CASE:
    if args.data_set == 'MNIST':
        if is_train:
            return transforms.Compose([
                transforms.Resize(args.input_size),
                # code expects 3 channel input
                transforms.Grayscale(num_output_channels=3),
                transforms.RandomCrop(args.input_size, padding=4),
                transforms.RandomRotation(10),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), 
                                     (0.3081, 0.3081, 0.3081))
            ])
        else: # eval MNIST
            return transforms.Compose([
                transforms.Resize(args.input_size),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize((0.1307, 0.1307, 0.1307), 
                                     (0.3081, 0.3081, 0.3081))
            ])
    
    # GENERAL CASE:
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation=args.train_interpolation,
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(
                args.input_size, padding=4)
        return transform

    t = []
    if resize_im:
        size = int(args.input_size / args.eval_crop_ratio)
        t.append(
            transforms.Resize(size, interpolation=3),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    if args.data_set == 'CIFAR10':
        mean = (0.4914, 0.4822, 0.4465)
        stdev = (0.2470, 0.2435, 0.2616)
    elif args.data_set == 'CIFAR100':
        mean = (0.5071, 0.4867, 0.4408)
        stdev = (0.2675, 0.2565, 0.2761)
    else:
        mean = IMAGENET_DEFAULT_MEAN
        stdev = IMAGENET_DEFAULT_STD
    t.append(transforms.Normalize(mean, stdev))
    return transforms.Compose(t)