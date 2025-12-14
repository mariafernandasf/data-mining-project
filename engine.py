"""
IEOR4540: I took this from https://github.com/naver-ai/rope-vit/tree/main/deit
"""

import math
import sys

import utils

from timm.data import Mixup
from timm.utils import accuracy, ModelEma

import torch

from typing import Iterable, Optional
import numpy as np

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    model_ema: Optional[ModelEma] = None, mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None, 
                    global_step = 0, 
                    sparse_learnable_variant = False,
                    lambda_l0 = 0.0,
                    r = 1e-5,
                    N_tau = 500):
    
    # set model to training mode
    model.train(set_training_mode)
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10

 
    i = 0
    for samples, targets in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # update gumbel sigmoid tau for sparse attention
        if (global_step % N_tau == 0) and sparse_learnable_variant:
            tau = max(0.5, math.exp(-r * global_step))
            for blk in model.blocks:
                if hasattr(blk, "attn"):
                    blk.attn.set_tau(tau)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)
            
            
        if args.bce_loss:
            targets = targets.gt(0.0).type(targets.dtype)
         
        with torch.amp.autocast('cuda'):
            outputs = model(samples)
            task_loss = criterion(outputs, targets)
            sparsity_loss = 0.0

            if sparse_learnable_variant:
                
                # sum L0 penalties over all attention blocks
                for blk in model.blocks:
                    if hasattr(blk, "attn"):
                        sparsity_loss = sparsity_loss + blk.attn.l0_penalty()
                
                loss = task_loss + (lambda_l0 * sparsity_loss)
            else:
                loss = task_loss

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        i += 1
        optimizer.zero_grad()

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()
        if model_ema is not None:
            model_ema.update(model)

        metric_logger.update(loss=loss_value)
        metric_logger.update(task_loss=task_loss.item())
        metric_logger.update(sparsity_loss=sparsity_loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

        global_step += 1
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, global_step


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for images, target in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 5))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}