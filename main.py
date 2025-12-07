"""
IEOR4540: This script is based on main.py from https://github.com/naver-ai/rope-vit/tree/main/deit

"""

import datetime
import json
import time
from pathlib import Path
import argparse 
import gc
import os

# imports from project
import const
from datasets import build_dataset
from engine import train_one_epoch, evaluate
import utils 
from augment import new_data_aug_generator
import models_v2
import models_v2_rope
import models_cayley
from samplers import RASampler

import numpy as np
import torch

# imports from timm
from timm.data import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.models import create_model
from timm.data import Mixup
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict
from timm.scheduler import create_scheduler

#from timm.scheduler import create_scheduler

def main(args):
    
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # pull datasets and number of output classes for the classification task
    dataset_train, args.nb_classes = build_dataset(is_train=True, args=args)
    dataset_val, _ = build_dataset(is_train=False, args=args)
    
    # ADD DISTRIBUTED CODE 
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        if args.repeated_aug:
            sampler_train = RASampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )
        if args.dist_eval:
            if len(dataset_val) % num_tasks != 0:
                print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                      'This will slightly alter validation results as extra duplicate entries are added to achieve '
                      'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False)
        else:
            sampler_val = torch.utils.data.SequentialSampler(dataset_val)
            # END DISTRIBUTED CODE 
    else: 
        # not running using distributed: 
        sampler_train = torch.utils.data.RandomSampler(dataset_train)
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
    )
    if args.ThreeAugment:
        data_loader_train.dataset.transform = new_data_aug_generator(args)

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False
    )

    # enable Mixup for data augmentation
    mixup_fn = None
    mixup_active = args.cutmix > 0. or args.cutmix_minmax is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup, cutmix_alpha=args.cutmix, cutmix_minmax=args.cutmix_minmax,
            prob=args.mixup_prob, switch_prob=args.mixup_switch_prob, mode=args.mixup_mode,
            label_smoothing=args.smoothing, num_classes=args.nb_classes)


    print(f"Creating model: {args.model}")
    model = create_model(
        args.model,
        pretrained=False, # init model with random weights
        num_classes=args.nb_classes, # num output classes for classification
        drop_rate=args.drop,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        img_size=args.input_size
    )

    # finetune
    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu', weights_only = False)
        checkpoint_model = checkpoint['model']
        
        state_dict = model.state_dict()
        for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        for k in ['freqs_t_x', 'freqs_t_y']:
            if k in checkpoint_model:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
                
        # interpolate position embedding
        if 'pos_embed' in checkpoint_model:
            pos_embed_checkpoint = checkpoint_model['pos_embed']
            embedding_size = pos_embed_checkpoint.shape[-1]
            num_patches = model.patch_embed.num_patches
            num_extra_tokens = model.pos_embed.shape[-2] - num_patches
            # height (== width) for the checkpoint position embedding
            orig_size = int((pos_embed_checkpoint.shape[-2] - num_extra_tokens) ** 0.5)
            # height (== width) for the new position embedding
            new_size = int(num_patches ** 0.5)
            # class_token and dist_token are kept unchanged
            extra_tokens = pos_embed_checkpoint[:, :num_extra_tokens]
            # only the position tokens are interpolated
            pos_tokens = pos_embed_checkpoint[:, num_extra_tokens:]
            pos_tokens = pos_tokens.reshape(-1, orig_size, orig_size, embedding_size).permute(0, 3, 1, 2)
            pos_tokens = torch.nn.functional.interpolate(
                pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
            pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2)
            new_pos_embed = torch.cat((extra_tokens, pos_tokens), dim=1)
            checkpoint_model['pos_embed'] = new_pos_embed

        model.load_state_dict(checkpoint_model, strict=False)
        
    model.to(device)

    model_ema = None
    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)

    if not args.unscale_lr:
        linear_scaled_lr = args.lr * args.batch_size * utils.get_world_size() / 512.0
        args.lr = linear_scaled_lr

    # create optimizer and loss_scaler
    optimizer = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    # init learning rate scheduler
    lr_scheduler, _ = create_scheduler(args, optimizer)


    # loss function
    criterion = LabelSmoothingCrossEntropy()

    if mixup_active:
        # smoothing is handled with mixup label transform
        criterion = SoftTargetCrossEntropy()
    elif args.smoothing:
        criterion = LabelSmoothingCrossEntropy(smoothing=args.smoothing)
    else:
        criterion = torch.nn.CrossEntropyLoss()
        
    if args.bce_loss:
        criterion = torch.nn.BCEWithLogitsLoss()

    teacher_model = None

    output_dir = Path(args.output_dir)

    # RESUME FROM CHECKPOINT 
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)

    # PERFORM EVAL ONLY 
    if args.eval:
        start_time = time.time()
        test_stats = evaluate(data_loader_val, model, device)
        
        # inference time
        eval_time = time.time() - start_time
        eval_time_str = str(datetime.timedelta(seconds=int(eval_time)))
        
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if utils.is_main_process():
            with (output_dir/'eval.txt').open("a") as f:
                f.write(json.dumps({"num_test_images": len(dataset_val),
                                    "acc1": f"{test_stats['acc1']:.1f}%",
                                    "acc5": f"{test_stats['acc5']:.1f}%",
                                    "loss": test_stats["loss"],
                                    "eval_time_str": eval_time_str,
                                    "eval_time_seconds": eval_time,
                                    "input_size": args.input_size, 
                                    # this assumes that the checkpoint path I am pulling was trained by
                                    # the start_epoch and epochs parameters in args
                                    "start_epoch": args.start_epoch,
                                    "epochs": args.epochs
                                }))
                f.write("\n")
        return
    

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(args.start_epoch, args.epochs):
        # distributed code
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        # end distributed code
        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad, model_ema, mixup_fn,
            set_training_mode=args.train_mode,  # keep in eval mode for deit finetuning / train mode for training and deit III finetuning
            args = args,
        )

        # update learning rate
        lr_scheduler.step(epoch)
        checkpoint_paths = [output_dir / 'checkpoint.pth']
        for checkpoint_path in checkpoint_paths:
            utils.save_on_master({
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                #'model_ema': get_state_dict(model_ema),
                'scaler': loss_scaler.state_dict(),
                'args': args,
            }, checkpoint_path)
                

        test_stats = evaluate(data_loader_val, model, device)
        print(f"Accuracy of the network on the {len(dataset_val)} test images: {test_stats['acc1']:.1f}%")
        
        if max_accuracy < test_stats["acc1"]:
            max_accuracy = test_stats["acc1"]

            checkpoint_paths = [output_dir / 'best_checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                    #'model_ema': get_state_dict(model_ema),
                    'scaler': loss_scaler.state_dict(),
                    'args': args,
                }, checkpoint_path)
            
        print(f'Max accuracy: {max_accuracy:.2f}%')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'test_{k}': v for k, v in test_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        
        
        
        
        if utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))
    if utils.is_main_process():
        with (output_dir / 'training_time.txt').open("a") as f:
            f.write(json.dumps({"start_epoch": args.start_epoch,
                                "end_epoch": args.epochs,
                                "training_time_str": total_time_str,
                                "training_time_seconds": total_time}))
            f.write("\n")
 
if __name__ == '__main__':
    torch.cuda.empty_cache()
    args = const.ARGS

    # add model name and dataset to output directory
    args["output_dir"] = args["output_dir"] + args["data_set"] + "/" + args["model"]
    print('\n\nOutput dir: {}\n\n'.format(args["output_dir"]))

    Path(args["output_dir"]).mkdir(parents=True, exist_ok=True)

    # set parameters that change depending on eval/ train to avoid human error
    if args["eval"]:
        args["finetune"] = args["output_dir"] + "/checkpoint.pth"
        with (Path(args["output_dir"] + "/eval_params.txt")).open("a") as f:
            f.write(json.dumps(args))
            f.write("\n")
    else:
        args["finetune"] = ""
        with (Path(args["output_dir"] + "/train_params.txt")).open("a") as f:
            f.write(json.dumps(args))
            f.write("\n")

    args = argparse.Namespace(**args)
    main(args)