import argparse
import math
import json
import os

import torch
import torch.nn.functional as F
import torch.distributed as dist

from utils import setup_for_distributed, MetricLogger, SmoothedValue, load_model, save_model
import models_adapter
from video_dataset import VideoDataset
from configs import DATASETS


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, required=True,
      help='model architecture name.')

  parser.add_argument('--batch_size', type=int, default=16,
      help='batch size per gpu')
  parser.add_argument('--blr', type=float, default=1e-3,
      help='base learning rate per 256 samples. actual base learning rate is linearly scaled '
           'based on batch size.')
  parser.add_argument('--lr', type=float,
      help='constant base learning rate. overrides the --blr option.')
  parser.add_argument('--weight_decay', type=float, default=1e-2,
      help='optimizer weight decay.')
  parser.add_argument('--epochs', type=int, default=10,
      help='number of training epochs.')
  parser.add_argument('--warmup_epochs', type=int, default=2,
      help='number of warmup epochs.')
  parser.add_argument('--eval_only', action='store_true',
      help='only run evaluation.')

  parser.add_argument('--save_dir', type=str,
      help='directory to save the checkpoints in. if empty no checkpoints are saved.')
  parser.add_argument('--auto_resume', action='store_true',
      help='automatically resume from the last checkpoint.')
  parser.add_argument('--auto_remove', action='store_true',
      help='automatically remove old checkpoint after generating a new checkpoint.')
  parser.add_argument('--save_freq', type=int, default=1,
      help='save checkpoint every n epochs.')
  parser.add_argument('--resume', type=str,
      help='manually specify checkpoint to resume from. overrides --auto_resume and --pretrain.')
  parser.add_argument('--pretrain', type=str,
      help='initialize model from the given checkpoint, discard mismatching weights and '
           'do not load optimizer states.')

  parser.add_argument('--dataset', type=str, required=True, choices=DATASETS.keys(),
      help='name of the dataset. the dataset should be configured in config.py.')
  parser.add_argument('--mirror', action='store_true',
      help='whether mirror augmentation (i.e., random horizontal flip) should be used during training.')
  parser.add_argument('--spatial_size', type=int, default=224,
      help='spatial crop size.')
  parser.add_argument('--num_frames', type=int, default=8,
      help='number of sampled frames per video.')
  parser.add_argument('--sampling_rate', type=int, default=0,
      help='interval between sampled frames. 0 means frames evenly covers the whole video '
           '(i.e., with variable frame interval depending on the video length).)')
  parser.add_argument('--num_spatial_views', type=int, default=1, choices=[1, 3],
      help='number of spatial crops used for testing (only 1 and 3 supported currently).')
  parser.add_argument('--num_temporal_views', type=int, default=1,
      help='number of temporal crops used for testing.')
  parser.add_argument('--auto_augment', type=str,
      help='enable RandAugment of a certain configuration. see the examples in the SSv2 training scripts.')
  parser.add_argument('--num_workers', type=int, default=16,
      help='number of dataloader workers.')
  parser.add_argument('--resize_type', type=str, default='random_resized_crop',
      choices=['random_resized_crop', 'random_short_side_scale_jitter'],
      help='spatial resize type. supported modes are "random_resized_crop" and "random_short_side_scale_jitter".'
           'see implementation in video_dataset/transform.py for the details.')
  parser.add_argument('--scale_range', type=float, nargs=2, default=[0.08, 1.0],
      help='range of spatial random resize. for random_resized_crop, the range limits the portion of the cropped area; '
           'for random_short_side_scale_jitter, the range limits the target short side (as the multiple of --spatial_size).')
  parser.add_argument('--print_freq', type=int, default=10, metavar='N',
      help='print a log message every N training steps.')
  parser.add_argument('--eval_freq', type=int, default=1, metavar='N',
      help='evaluate on the validation set every N epochs.')

  args = parser.parse_args()

  dist.init_process_group('nccl')
  gpu_id = dist.get_rank() % torch.cuda.device_count()
  torch.cuda.set_device(gpu_id)
  setup_for_distributed(dist.get_rank() == 0)

  print("{}".format(args).replace(', ', ',\n'))

  print('creating model')
  model = models_adapter.__dict__[args.model](num_classes=DATASETS[args.dataset]['NUM_CLASSES']).cuda().train()
  n_trainable_params = 0
  for n, p in model.named_parameters():
    if p.requires_grad:
      print('Trainable param: %s, %s, %s' % (n, p.size(), p.dtype))
      n_trainable_params += p.numel()
  print('Total trainable params:', n_trainable_params, '(%.2f M)' % (n_trainable_params / 1000000))
  model = torch.nn.parallel.DistributedDataParallel(model)
  model_without_ddp = model.module

  print('creating dataset')
  if not args.eval_only:
    dataset_train = VideoDataset(
        list_path=DATASETS[args.dataset]['TRAIN_LIST'],
        data_root=DATASETS[args.dataset]['TRAIN_ROOT'],
        random_sample=True,
        mirror=args.mirror,
        spatial_size=args.spatial_size,
        auto_augment=args.auto_augment,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        resize_type=args.resize_type,
        scale_range=args.scale_range,
        )
    print('train dataset:', dataset_train)
  dataset_val = VideoDataset(
      list_path=DATASETS[args.dataset]['VAL_LIST'],
      data_root=DATASETS[args.dataset]['VAL_ROOT'],
      random_sample=False,
      spatial_size=args.spatial_size,
      num_frames=args.num_frames,
      sampling_rate=args.sampling_rate,
      num_spatial_views=args.num_spatial_views,
      num_temporal_views=args.num_temporal_views,
      )
  print('val dataset:', dataset_val)

  if not args.eval_only:
    dataloader_train = torch.utils.data.DataLoader(
        dataset_train,
        batch_size=args.batch_size,
        sampler=torch.utils.data.DistributedSampler(dataset_train),
        num_workers=args.num_workers,
        pin_memory=True,
        )
  dataloader_val = torch.utils.data.DataLoader(
      torch.utils.data.Subset(dataset_val, range(dist.get_rank(), len(dataset_val), dist.get_world_size())),
      batch_size=1,
      shuffle=False,
      num_workers=args.num_workers,
      pin_memory=True,
      )

  if args.eval_only:
    optimizer = None
    loss_scaler = None
    lr_sched = None
  else:
    if args.lr is not None:
      print('using absolute lr:', args.lr)
    else:
      print('using relative lr (per 256 samples):', args.blr)
      args.lr = args.blr * args.batch_size * dist.get_world_size() / 256
      print('effective lr:', args.lr)
    
    params_with_decay, params_without_decay = [], []
    for n, p in model.named_parameters():
      if not p.requires_grad:
        continue
      if '.bias' in n:
        params_without_decay.append(p)
      else:
        params_with_decay.append(p)
    optimizer = torch.optim.AdamW(
        [
          {'params': params_with_decay, 'lr': args.lr, 'weight_decay': args.weight_decay},
          {'params': params_without_decay, 'lr': args.lr, 'weight_decay': 0.}
        ],
        )
    print(optimizer)
    loss_scaler = torch.cuda.amp.GradScaler()
    
    def lr_func(step):
      epoch = step / len(dataloader_train)
      if epoch < args.warmup_epochs:
        return epoch / args.warmup_epochs
      else:
        return 0.5 + 0.5 * math.cos((epoch - args.warmup_epochs) / (args.epochs - args.warmup_epochs) * math.pi)
    lr_sched = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_func)

  def evaluate(log_stats=None):
    metric_logger = MetricLogger(delimiter="  ")
    header = 'Test:'
    model.eval()
    for data, labels in metric_logger.log_every(dataloader_val, 100, header):
      data, labels = data.cuda(), labels.cuda()
      B, V = data.size(0), data.size(1)
      data = data.flatten(0, 1)
      with torch.cuda.amp.autocast():
        with model.no_sync():
          with torch.no_grad():
            logits = model(data)
        scores = logits.softmax(dim=-1)
        scores = scores.view(B, V, -1).mean(dim=1)
        acc1 = (scores.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
        acc5 = (scores.topk(5, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
      metric_logger.meters['acc1'].update(acc1, n=scores.size(0))
      metric_logger.meters['acc5'].update(acc5, n=scores.size(0))
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f}'
        .format(top1=metric_logger.acc1, top5=metric_logger.acc5))
    if log_stats is not None:
      log_stats.update({'val_' + k: meter.global_avg for k, meter in metric_logger.meters.items()})

  start_epoch = load_model(args, model_without_ddp, optimizer, lr_sched, loss_scaler)

  if args.eval_only:
    evaluate()
    return

  for epoch in range(start_epoch, args.epochs):
    dataloader_train.sampler.set_epoch(epoch)
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    model.train()
    for step, (data, labels) in enumerate(metric_logger.log_every(dataloader_train, args.print_freq, header)):
      data, labels = data.cuda(), labels.cuda()
      optimizer.zero_grad()
      with torch.cuda.amp.autocast():
        logits = model(data)
        acc1 = (logits.topk(1, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
        acc5 = (logits.topk(5, dim=1)[1] == labels.view(-1, 1)).sum(dim=-1).float().mean().item() * 100
        loss = F.cross_entropy(logits, labels)
      loss_scaler.scale(loss).backward()
      loss_scaler.step(optimizer)
      lr_sched.step()
      loss_scaler.update()

      metric_logger.update(
          loss=loss.item(),
          lr=optimizer.param_groups[0]['lr'],
          acc1=acc1, acc5=acc5,
          )

    print('Averaged stats:', metric_logger)
    log_stats = {'train_' + k: meter.global_avg for k, meter in metric_logger.meters.items()}

    save_model(args, epoch, model_without_ddp, optimizer, lr_sched, loss_scaler)

    if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == args.epochs:
      evaluate(log_stats)

    if args.save_dir is not None and dist.get_rank() == 0:
      n_total_params, n_trainable_params = 0, 0
      for n, p in model_without_ddp.named_parameters():
        n_total_params += p.numel()
        if p.requires_grad:
          n_trainable_params += p.numel()
      log_stats['epoch'] = epoch
      log_stats['n_trainable_params'] = n_trainable_params
      log_stats['n_total_params'] = n_total_params
      with open(os.path.join(args.save_dir, 'log.txt'), 'a') as f:
        f.write(json.dumps(log_stats) + '\n')


if __name__ == '__main__': main()
