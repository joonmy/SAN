from pickletools import optimize
# from sched import scheduler
import torch
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler as scheduler
from torch.nn.utils.rnn import pad_sequence
from torch.nn import functional as F
from torch import nn
from torch.utils.data import DataLoader, SequentialSampler
import torch.distributed as dist

from utils import CrossEn

# *transformers
from transformers import AutoTokenizer, MBartForConditionalGeneration, MBartTokenizer,MBartConfig, BertTokenizer, BertModel, BertTokenizerFast,MBart50TokenizerFast, MBartTokenizerFast

# *user-defined
from models import  SLRCLIP, Text_Decoder
import utils as utils
from datasets import S2T_Dataset

# *basic
import os
import time
import shutil
import argparse, json, datetime
import numpy as np
from collections import OrderedDict
from tqdm import tqdm
import yaml
import random
import wandb
import copy
from pathlib import Path
import math
import sys
from typing import Iterable, Optional
from loguru import logger


# *metric
from sacrebleu.metrics import BLEU, CHRF, TER

# *timm
from timm.optim import create_optimizer
from timm.scheduler import create_scheduler
from timm.utils import NativeScaler
from timm.loss import SoftTargetCrossEntropy
from timm.optim import AdamW

# visualization
from torchvision.utils import save_image, make_grid
from PIL import Image

from hpman.m import _
import hpargparse


# global definition
from definition import *

def get_args_parser():
    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', add_help=False)
    parser.add_argument('--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=80, type=int)
    
    parser.add_argument('--neg_table_name', default='', type=str)
    parser.add_argument('--num_hard', default=5, type=int)


    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--local_rank', default=0, type=int)


    # * Finetuning params
    parser.add_argument('--finetune', default='', help='finetune from checkpoint')

    # * Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt-eps', default=1.0e-09, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1.0e-09)')
    parser.add_argument('--opt-betas', default=None, type=float, nargs='+', metavar='BETA',
                        help='Optimizer Betas (default: [0.9, 0.98], use opt default)')
    parser.add_argument('--clip-grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=0.001,
                        help='weight decay (default: 0.05)')

    # * Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=1.0e-3, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--lr-noise', type=float, nargs='+', default=None, metavar='pct, pct',
                        help='learning rate noise on/off epoch percentages')
    parser.add_argument('--lr-noise-pct', type=float, default=0.67, metavar='PERCENT',
                        help='learning rate noise limit percent (default: 0.67)')
    parser.add_argument('--lr-noise-std', type=float, default=1.0, metavar='STDDEV',
                        help='learning rate noise std-dev (default: 1.0)')
    parser.add_argument('--warmup-lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min-lr', type=float, default=1.0e-08, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    
    parser.add_argument('--decay-epochs', type=float, default=30, metavar='N',
                        help='epoch interval to decay LR')
    parser.add_argument('--warmup-epochs', type=int, default=0, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown-epochs', type=int, default=10, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--patience-epochs', type=int, default=10, metavar='N',
                        help='patience epochs for Plateau LR scheduler (default: 10')
    parser.add_argument('--decay-rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')
    
     # * Baise params
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True)
    parser.add_argument('--config', type=str, default='./configs/config_gloss_free.yaml')

    # * data process params
    parser.add_argument('--input-size', default=224, type=int)
    parser.add_argument('--resize', default=256, type=int)
    
    # * wandb params
    parser.add_argument("--log_all", action="store_true",
        help="flag to log in all processes, otherwise only in rank0",
    )
    parser.add_argument("--entity", type=str, 
        help="wandb entity",
    )
    parser.add_argument("--project", type=str, default='Thesis',
        help="wandb project",
    )

    parser.add_argument('--noise-rate', default=0.15, type=float)
    parser.add_argument('--noise-type', default='omit_last', type=str, choices=['omit', 'omit_last'])
    parser.add_argument('--random-shuffle', default=False, type=bool)

    parser.add_argument('--loss_lambda', type=float, default=0, metavar='RATE',
                        help='lambda param')

    return parser

def main(args, config):

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True 

    print(f"Creating dataset:")
    tokenizer = BertTokenizerFast.from_pretrained("google-bert/bert-base-german-cased")
    train_data = S2T_Dataset(path=config['data']['train_label_path'], tokenizer = tokenizer, config=config, args=args, phase='train')
    print(train_data)
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_data,shuffle=True)
    train_dataloader = DataLoader(train_data,
                                 batch_size=args.batch_size, 
                                 num_workers=args.num_workers, 
                                 collate_fn=train_data.collate_fn,
                                 sampler=train_sampler, 
                                 pin_memory=args.pin_mem,
                                 drop_last=True)

    test_data = S2T_Dataset(path=config['data']['test_label_path'], tokenizer = tokenizer, config=config, args=args, phase='test')
    print(test_data)

    test_dataloader = DataLoader(
        test_data,
        batch_size=256,
        num_workers=args.num_workers,
        collate_fn=test_data.collate_fn,
        pin_memory=args.pin_mem,
    )

    print(f"Creating model:")
    model = SLRCLIP(config=config, args=args)
    model.to(device)
    print(model)

    if args.finetune:
        checkpoint = torch.load(args.finetune, map_location='cpu')
        ret =  model.load_state_dict(checkpoint['model'], strict=False)
        print('Missing keys: \n', '\n'.join(ret.missing_keys))
        print('Unexpected keys: \n', '\n'.join(ret.unexpected_keys))

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], find_unused_parameters=True)
        model_without_ddp = model.module
    n_parameters = utils.count_parameters_in_MB(model_without_ddp)
    print(f'number of params: {n_parameters}M')


    optimizer = create_optimizer(args, model_without_ddp)
    lr_scheduler, _ = create_scheduler(args, optimizer)
    print(optimizer)
    print(lr_scheduler)

    criterion = CrossEn()
    loss_scaler = NativeScaler()

    output_dir = Path(args.output_dir)
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1

    if args.eval:
        if not args.resume:
            logger.warning('Please specify the trained model: --resume /path/to/best_checkpoint.pth')

        test_stats, _, _, _ = evaluate(args, test_dataloader, model, criterion, args.start_epoch)
        print(f"Test loss of the network on the {len(test_dataloader)} test videos: {test_stats['loss']:.3f}")
        return

    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    dev_min_loss = np.inf
    dev_best_avg = 0
    dev_best_r1_t2i = 0
    dev_best_mrr_t2i = 0

    for epoch in range(args.start_epoch, args.epochs):
        
        if args.distributed:
            train_dataloader.sampler.set_epoch(epoch)
        
        train_stats = train_one_epoch(args, model, criterion, train_dataloader, optimizer, device, epoch, config, loss_scaler)
        lr_scheduler.step(epoch)

        if args.output_dir:
            checkpoint_paths = [output_dir / f'checkpoint.pth']
            for checkpoint_path in checkpoint_paths:
                utils.save_on_master({
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'epoch': epoch,
                }, checkpoint_path)

        if utils.is_main_process():
            dev_avg, dev_r1_t2i, dev_mrr_t2i = evaluate(args, test_dataloader, model, criterion, epoch)

            if dev_best_avg < dev_avg:
                dev_best_avg = dev_avg
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'dev_best_score_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            # 'args': args,
                        }, checkpoint_path)
            if dev_best_r1_t2i < dev_r1_t2i:
                dev_best_r1_t2i = dev_r1_t2i
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'dev_best_r1_t2i_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            # 'args': args,
                        }, checkpoint_path)
            if dev_best_mrr_t2i < dev_mrr_t2i:
                dev_best_mrr_t2i = dev_mrr_t2i
                if args.output_dir:
                    checkpoint_paths = [output_dir / 'dev_best_mrr_t2i_checkpoint.pth']
                    for checkpoint_path in checkpoint_paths:
                        utils.save_on_master({
                            'model': model_without_ddp.state_dict(),
                            'optimizer': optimizer.state_dict(),
                            'lr_scheduler': lr_scheduler.state_dict(),
                            'epoch': epoch,
                            # 'args': args,
                        }, checkpoint_path)
        
            if args.run:
                args.run.log({'epoch':epoch+1,'training/train_loss':train_stats['loss']})

    test_on_last_epoch = True
    if test_on_last_epoch and args.output_dir and utils.is_main_process():
        torch.distributed.barrier()
        print("last evaluation on dev and test set!!!")
        checkpoint = torch.load(args.output_dir+'/dev_best_score_checkpoint.pth', map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'], strict=True)

        _, _, _ = evaluate(args, test_dataloader, model, criterion, epoch, last=True)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def train_one_epoch(args, model: torch.nn.Module, criterion: nn.CrossEntropyLoss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, config, loss_scaler, max_norm: float = 0,
                    set_training_mode=True):
    model.train(set_training_mode)

    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    print_freq = 10
    loss_crossen = criterion
    loss_txt = criterion

    for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            logits_per_image, logits_per_text, hard_i2t_sim = model(src_input, tgt_input)
            loss_imgs = loss_crossen(logits_per_image)
            loss_texts = loss_crossen(logits_per_text)
            loss_hard = loss_crossen(hard_i2t_sim, is_hard=True)
            total_loss = (loss_imgs + loss_texts)/2. + args.loss_lambda * loss_hard
        loss_scaler(total_loss, optimizer)

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(loss_hard=loss_hard.item())

    if args.run:
        args.run.log({'epoch':epoch+1,'epoch/train_loss':loss_value})

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    return  {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def evaluate(args, dev_dataloader, model, criterion, epoch, last=False, test=False):
    model.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    print_freq = 10
    loss_img = criterion
    loss_txt = criterion

    image_feat_list = []
    text_feat_list = []
    image_mask_list = []
    text_mask_list = []

    with torch.no_grad():
        for step, (src_input, tgt_input) in enumerate(metric_logger.log_every(dev_dataloader, print_freq, header)):

            image_features, text_features, image_masks, text_masks = model(src_input, tgt_input, eval_mode=True)
            image_feat_list.append(image_features)
            
            text_feat_list.append(text_features)
            image_mask_list.append(image_masks)
            text_mask_list.append(text_masks)

    print("eval!!")
    all_image_features = torch.cat(image_feat_list, dim=0)
    all_text_features = torch.cat(text_feat_list, dim=0)
    all_image_masks = torch.cat(image_mask_list, dim=0)
    all_text_masks = torch.cat(text_mask_list, dim=0)
    print("all_image_features shape:", all_image_features.shape)
    print("all_text_features shape:", all_text_features.shape)
    print("all_image_masks shape:", all_image_masks.shape)
    print("all_text_masks shape:", all_text_masks.shape)

    with torch.no_grad():
        i2t_sim = torch.einsum('ais, bjs -> abij', all_image_features, all_text_features) # [B*word_size, B*word_size, T, T']
        t2i_sim = torch.einsum('ais, bjs -> abij', all_text_features, all_image_features) # [B*word_size, B*word_size, T', T]
    
        after_softmax_i2t = torch.nansum(i2t_sim * torch.softmax(i2t_sim/0.07, dim=-1), dim=-1) # [B*word_size, B*word_size, T]
        image_mask_extended = all_image_masks.unsqueeze(1).repeat(1, all_image_masks.shape[0], 1) # [B*word_size, B*word_size, T]
        after_softmax_i2t[~image_mask_extended] = 0
        I2T_sim = model.module.logit_scale * torch.nansum(after_softmax_i2t, dim=-1) / torch.sum(image_mask_extended, dim=-1)

        after_softmax_t2i = torch.nansum(t2i_sim * torch.softmax(t2i_sim/0.07, dim=-1), dim=-1) # [B*word_size, B*word_size, T']
        text_mask_extended = all_text_masks.unsqueeze(1).repeat(1, all_text_masks.shape[0], 1) # [B*word_size, B*word_size, T']
        after_softmax_t2i[~text_mask_extended] = 0
        T2I_sim = model.module.logit_scale * torch.nansum(after_softmax_t2i, dim=-1) / torch.sum(text_mask_extended, dim=-1)
    
    print("I2T_sim shape:", I2T_sim.shape)
    print("T2I_sim shape:", T2I_sim.shape)

    N = I2T_sim.shape[0]
    # ground-truth: image i matches text i
    ranks_i2t = torch.argsort(I2T_sim, dim=-1, descending=True)
    ranks_t2i = torch.argsort(T2I_sim, dim=-1, descending=True)
    # image->text
    hits1_i2t = (ranks_i2t[:, :1] == torch.arange(N, device=ranks_i2t.device).unsqueeze(1)).any(dim=1).float()
    hits5_i2t = (ranks_i2t[:, :5] == torch.arange(N, device=ranks_i2t.device).unsqueeze(1)).any(dim=1).float()
    hits10_i2t = (ranks_i2t[:, :10] == torch.arange(N, device=ranks_i2t.device).unsqueeze(1)).any(dim=1).float()
    # text->image
    hits1_t2i = (ranks_t2i[:, :1] == torch.arange(N, device=ranks_t2i.device).unsqueeze(1)).any(dim=1).float()
    hits5_t2i = (ranks_t2i[:, :5] == torch.arange(N, device=ranks_t2i.device).unsqueeze(1)).any(dim=1).float()
    hits10_t2i = (ranks_t2i[:, :10] == torch.arange(N, device=ranks_t2i.device).unsqueeze(1)).any(dim=1).float()

    gt_indices = torch.arange(N, device=ranks_i2t.device)
    ranks_correct_i2t = (ranks_i2t == gt_indices.unsqueeze(1)).nonzero()[:, 1]
    print("ranks_correct_i2t", ranks_correct_i2t)
    reciprocal_ranks_i2t = 1.0 / (ranks_correct_i2t.float() + 1.0)
    mrr_i2t = reciprocal_ranks_i2t.mean().item()

    ranks_correct_t2i = (ranks_t2i == gt_indices.unsqueeze(1)).nonzero()[:, 1]
    reciprocal_ranks_t2i = 1.0 / (ranks_correct_t2i.float() + 1.0)
    mrr_t2i = reciprocal_ranks_t2i.mean().item()

    r1_i2t = hits1_i2t.mean().item()
    r5_i2t = hits5_i2t.mean().item()
    r10_i2t = hits10_i2t.mean().item()
    r1_t2i = hits1_t2i.mean().item()
    r5_t2i = hits5_t2i.mean().item()
    r10_t2i = hits10_t2i.mean().item()

    avg = r1_t2i + r5_t2i + r10_t2i 

    if args.run and not last and not test:
        args.run.log({
            'epoch': epoch + 1,
            'dev/R1_i2t': r1_i2t,
            'dev/R5_i2t': r5_i2t,
            'dev/R10_i2t': r10_i2t,
            'dev/R1_t2i': r1_t2i,
            'dev/R5_t2i': r5_t2i,
            'dev/R10_t2i': r10_t2i,
            'dev/mrr_i2t': mrr_i2t,
            'dev/mrr_t2i': mrr_t2i,
        })

    print('* mrr_t2i {mrr_t2i:.4f} R1_t2i {r1_t2i:.4f} R5_t2i {r5_t2i:.4f} R10_t2i {r10_t2i:.4f} mrr_i2t {mrr_i2t:.4f} R1_i2t {r1_i2t:.4f} R5_i2t {r5_i2t:.4f} R10_i2t {r10_i2t:.4f}'.format(
        r1_i2t=r1_i2t, r5_i2t=r5_i2t, r10_i2t=r10_i2t, r1_t2i=r1_t2i, r5_t2i=r5_t2i, r10_t2i=r10_t2i, mrr_i2t=mrr_i2t, mrr_t2i=mrr_t2i))

    del image_feat_list, text_feat_list, image_mask_list, text_mask_list, all_image_features, all_text_features, all_image_masks, all_text_masks, i2t_sim, t2i_sim, I2T_sim, T2I_sim
    torch.cuda.empty_cache()
    
    return avg, r1_t2i, mrr_t2i

def setup_run(args, config):
    if args.log_all:
        os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            group=args.output_dir.split('/')[-1],
            config=config,
        )
        run.define_metric("epoch")
        run.define_metric("training/*", step_metric="epoch")
        run.define_metric("dev/*", step_metric="epoch")
    else:
        if utils.is_main_process():
            os.environ["WANDB_MODE"] = config['training']['wandb'] if not args.eval else 'disabled'
            run = wandb.init(
                entity=args.entity,
                project=args.project,
                config=config,
            )
            run.define_metric("epoch")
            run.define_metric("training/*", step_metric="epoch")
            run.define_metric("dev/*", step_metric="epoch")
            run.name = args.output_dir.split('/')[-1]
        else:
            os.environ["WANDB_MODE"] = 'disabled'
            run = False

    return run
if __name__ == '__main__':

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser('Visual-Language-Pretraining (VLP) V2 scripts', parents=[get_args_parser()])
    _.parse_file(Path(__file__).resolve().parent)
    hpargparse.bind(parser, _)
    args = parser.parse_args()

    with open(args.config, 'r+',encoding='utf-8') as f:
        config = yaml.load(f,Loader=yaml.FullLoader)

    utils.init_distributed_mode(args)
    print(args)
    
    # wandb.init a run if logging, otherwise return None
    args.run = setup_run(args, config)
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    try:
        main(args, config)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
    finally:
        if args.run:
            args.run.finish()
        if args.distributed:
            print("Cleaning up process group")
            dist.barrier()
            dist.destroy_process_group()
    