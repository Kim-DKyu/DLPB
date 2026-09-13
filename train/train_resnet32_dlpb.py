import time
import datetime
import sys
from pathlib import Path

FINAL_DIR = Path(__file__).resolve().parent.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))
from collections import deque

import torch

from args import get_args_parser
from setup import setup
from dataset import get_dataset, get_dataloader
from models.resnet32_dlpb import create_resnet

# from resnet_Ga import create_resnet #이게 진짜임
from optimizer import get_optimizer_and_scheduler, get_scaler_criterion
from metric import Metric, Accuracy, reduce_mean
from utils import print_metadata, save_checkpoint, get_ema_ddp_model, resume_from_checkpoint
from log import Result
import torch.nn.functional as F


def fast_forward_dataloader_rng(train_dataloader, valid_dataloader, completed_epochs):
    """Restore epoch-boundary sampler and worker RNG streams for legacy checkpoints."""
    sampler_generator = getattr(train_dataloader.sampler, 'generator', None)
    if sampler_generator is not None:
        for _ in range(completed_epochs):
            deque(train_dataloader.sampler, maxlen=0)

    for dataloader in (train_dataloader, valid_dataloader):
        worker_generator = getattr(dataloader, 'generator', None)
        if worker_generator is not None:
            for _ in range(completed_epochs):
                torch.empty((), dtype=torch.int64).random_(generator=worker_generator)


@torch.inference_mode()
def validate(valid_dataloader, model, criterion, args, mode='org'):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    top1_m = Metric(reduce_every_n_step=args.print_freq, header='Top-1:')
    top5_m = Metric(reduce_every_n_step=args.print_freq, header='Top-5:')
    loss_m = Metric(reduce_every_n_step=args.print_freq, header='Loss:')

    # 2. start validate
    model.eval()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(valid_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(valid_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            outputs, y_hat = model(x)
            # outputs, y_hat = model(x)
            loss = criterion(y_hat, y)

        top1, top5 = Accuracy(y_hat, y, top_k=(1,5,))

        top1_m.update(top1, batch_size)
        top5_m.update(top5, batch_size)
        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"VALID({mode}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m} {top1_m} {top5_m}")

        batch_m.update(time.time() - start_time)
        start_time = time.time()


    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    top1 = top1_m.compute()
    top5 = top5_m.compute()
    loss = loss_m.compute()

    # 4. print metric
    space = 16
    num_metric = 6
    args.log('-'*space*num_metric)
    args.log(("{:>16}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Top-1 Acc', 'Top-5 Acc'))
    args.log('-'*space*num_metric)
    args.log(f"{'VALID('+mode+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{top1:{space}.4f}{top5:{space}.4f}")
    args.log('-'*space*num_metric)

    return loss, top1, top5


def train_one_epoch(train_dataloader, model, optimizer, criterion, args, ema_model=None, scheduler=None, scaler=None, epoch=None):
    # 1. create metric
    data_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Data:')
    batch_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Batch:')
    loss_m = Metric(reduce_every_n_step=0, reduce_on_compute=False, header='Loss:')

    # 2. start validate
    model.train()
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    total_iter = len(train_dataloader)
    start_time = time.time()

    for batch_idx, (x, y) in enumerate(train_dataloader):
        batch_size = x.size(0)
        x = x.to(args.device)
        y = y.to(args.device)

        if args.channels_last:
            x = x.to(memory_format=torch.channels_last)

        data_m.update(time.time() - start_time)

        with torch.cuda.amp.autocast(args.amp):
            outputs, y_hat = model(x)
            output_add = 0
            loss = 0

            ce_loss = args.alpha*criterion(y_hat, y)
            loss += ce_loss

            for out in outputs:
                loss += criterion(out, y)
                output_add += out.data

            for out_ in outputs:
                loss_distil_bw = F.kl_div(F.log_softmax(y_hat / args.temperature), F.log_softmax(
                    out_.detach() / args.temperature), reduction='batchmean', log_target=True) * (
                                          args.temperature ** 2)
                loss += loss_distil_bw

            loss_distil_ens = F.kl_div(F.log_softmax(y_hat / args.temperature), F.log_softmax(
                output_add.detach() / args.temperature), reduction='batchmean', log_target=True) * (
                                      args.temperature ** 2)
            loss += loss_distil_ens

            for i, out in enumerate(outputs):
                loss_dec = F.kl_div(F.log_softmax(out + 0), F.log_softmax(
                    (output_add.detach() / len(outputs)) + 0), reduction=args.dec_reduction, log_target=True) * args.GA_lamb
                loss += loss_dec

        if args.distributed:
            loss = reduce_mean(loss, args.world_size)

        if args.amp:
            scaler(loss, optimizer, model.parameters(), scheduler, args.grad_norm, batch_idx % args.grad_accum == 0)
        else:
            loss.backward()

            if args.grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            if batch_idx % args.grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler:
                    scheduler.step()

        loss_m.update(loss, batch_size)

        if batch_idx and args.print_freq and batch_idx % args.print_freq == 0:
            num_digits = len(str(total_iter))
            args.log(f"TRAIN({epoch:03}): [{batch_idx:>{num_digits}}/{total_iter}] {batch_m} {data_m} {loss_m}")

        if batch_idx and ema_model and batch_idx % args.ema_update_step == 0:
            ema_model.update(model)

        batch_m.update(time.time() - start_time)
        start_time = time.time()

    # 3. calculate metric
    duration = str(datetime.timedelta(seconds=batch_m.sum)).split('.')[0]
    data = str(datetime.timedelta(seconds=data_m.sum)).split('.')[0]
    f_b_o = str(datetime.timedelta(seconds=batch_m.sum - data_m.sum)).split('.')[0]
    loss = loss_m.compute()

    # 4. print metric
    space = 16
    num_metric = 5
    args.log('-'*space*num_metric)
    args.log(("{:>16}"*num_metric).format('Stage', 'Batch', 'Data', 'F+B+O', 'Loss'))
    args.log('-'*space*num_metric)
    args.log(f"{'TRAIN('+str(epoch)+')':>{space}}{duration:>{space}}{data:>{space}}{f_b_o:>{space}}{loss:{space}.4f}")
    args.log('-'*space*num_metric)

    return loss

def extract_fc_weight_matrix(model, layer_name="classifier"):
    for name, param in model.named_parameters():
        if layer_name in name and "weight" in name:
            return param.detach().cpu().numpy()
    return None
def run(args):
    # 0. init ddp & logger
    setup(args)

    # 1. load dataset
    train_dataset, valid_dataset = get_dataset(args)
    train_dataloader, valid_dataloader = get_dataloader(train_dataset, valid_dataset, args)

    # 2. make model
    model = create_resnet(args.model_name, args.num_classes)
    model, ema_model, ddp_model = get_ema_ddp_model(model, args)

    # 3. load optimizer
    optimizer, scheduler = get_optimizer_and_scheduler(model, args)

    # 4. load criterion
    criterion, valid_criterion, scaler = get_scaler_criterion(args)

    # 5. print metadata
    print_metadata(model, train_dataset, valid_dataset, args)

    # 6. control logic for checkpoint & validate
    if args.resume:
        start_epoch = resume_from_checkpoint(args.checkpoint_path, model, ema_model, optimizer, scaler, scheduler)
    else:
        start_epoch = 0

    start_epoch = args.start_epoch if args.start_epoch else start_epoch
    end_epoch = args.end_epoch if args.end_epoch else args.epoch

    if args.resume and start_epoch:
        fast_forward_dataloader_rng(train_dataloader, valid_dataloader, start_epoch)
        args.log(f'Restored data RNG streams for resume at epoch {start_epoch}.')

    if scheduler is not None and start_epoch and not args.resume:
        # Schedulers are stepped per optimizer update, not per epoch.
        scheduler.step(start_epoch * args.iter_per_epoch)

    if args.validate_only:
        validate(valid_dataloader, model, valid_criterion, args, 'org')
        if args.ema:
            validate(valid_dataloader, ema_model, valid_criterion, args, 'ema')
        return

    # 7. train
    best_epoch = 0
    best_acc = 0
    top1_list = []
    top5_list = []
    start_time = time.time()

    for epoch in range(start_epoch, end_epoch):
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)

        train_loss = train_one_epoch(train_dataloader, ddp_model if args.distributed else model, optimizer, criterion, args, ema_model, scheduler, scaler, epoch)
        val_loss, top1, top5 = validate(valid_dataloader, ddp_model if args.distributed else model, valid_criterion, args, 'org')
        if args.ema:
            eval_ema_metric = validate(valid_dataloader, ema_model.module, valid_criterion, args, 'ema')

        if args.use_wandb:
            args.log({'epoch':epoch, 'train_loss':train_loss, 'val_loss':val_loss, 'top1':top1, 'top5':top5}, metric=True)

        if best_acc < top1:
            best_acc = top1
            best_epoch = epoch
        top1_list.append(top1)
        top5_list.append(top5)

        if args.save_checkpoint and args.is_rank_zero:
            save_checkpoint(args.log_dir, model, ema_model, optimizer,
                            scaler, scheduler, epoch, is_best=best_epoch == epoch)

    # 8. summary train result in csv
    if args.is_rank_zero:
        best_acc = round(float(best_acc), 4)
        top1 = round(float(sum(top1_list[-3:]) / 3), 4)
        top5 = round(float(sum(top5_list[-3:]) / 3), 4)
        duration = str(datetime.timedelta(seconds=time.time() - start_time)).split('.')[0]
        Result(args.output_dir).save_result(args, top1_list, top5_list,
                                            dict(duration=duration, best_acc=best_acc, avg_top1_acc=top1, avg_top5_acc=top5))


if __name__ == '__main__':
    args_parser = get_args_parser()
    args = args_parser.parse_args()
    run(args)