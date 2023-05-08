# modified from https://github.com/facebookresearch/mae/blob/efb2a8062c206524e35e47d04501ed4f544c0ae8/util/misc.py#L170

import os
import builtins
import datetime
from collections import defaultdict, deque
import time

import torch
import torch.distributed as dist


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = builtins.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            now = datetime.datetime.now().time()
            builtin_print('[{}] '.format(now), end='')  # print with time stamp
            builtin_print(*args, **kwargs)

    builtins.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.4f} ({global_avg:.4f})"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0 
        self.count = 0 
        self.fmt = fmt 

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n 

    def synchronize_between_processes(self):
        """ 
        Warning: does not synchronize the deque!
        """
        if not is_dist_avail_and_initialized():
            return
        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        dist.barrier()
        dist.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if v is None:
                continue
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append(
                "{}: {}".format(name, str(meter))
            )   
        return self.delimiter.join(loss_str)

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def log_every(self, iterable, print_freq, header=None):
        i = 0 
        if not header:
            header = ''
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt='{avg:.4f}')
        data_time = SmoothedValue(fmt='{avg:.4f}')
        space_fmt = ':' + str(len(str(len(iterable)))) + 'd' 
        log_msg = [ 
            header,
            '[{0' + space_fmt + '}/{1}]',
            'eta: {eta}',
            '{meters}',
            'time: {time}',
            'data: {data}'
        ]   
        if torch.cuda.is_available():
            log_msg.append('max mem: {memory:.0f}')
        log_msg = self.delimiter.join(log_msg)
        MB = 1024.0 * 1024.0
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj 
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                eta_seconds = iter_time.global_avg * (len(iterable) - i)
                eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
                if torch.cuda.is_available():
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time),
                        memory=torch.cuda.max_memory_allocated() / MB))
                else:
                    print(log_msg.format(
                        i, len(iterable), eta=eta_string,
                        meters=str(self),
                        time=str(iter_time), data=str(data_time)))
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(
            header, total_time_str, total_time / len(iterable)))


def load_model(args, model_without_ddp, optimizer, lr_sched, loss_scaler):
  if args.resume is None and args.auto_resume:
    print('trying to auto resume from save directory')
    if os.path.isdir(args.save_dir):
      ckpts = [x for x in os.listdir(args.save_dir) if x.startswith('checkpoint-') and x.endswith('.pth')]
    else:
      ckpts = []
    ckpt_epochs = [int(x[len('checkpoint-'):-len('.pth')]) for x in ckpts]
    ckpt_epochs.sort()
    print(f'{len(ckpt_epochs)} candidate checkpoint(s) found.')
    for epoch in ckpt_epochs[::-1]:
      ckpt_path = os.path.join(args.save_dir, 'checkpoint-%d.pth' % epoch)
      try:
        torch.load(ckpt_path, map_location='cpu')
      except Exception as e:
        print(f'loading checkpoint {ckpt_path} failed with error:', e)
        continue
      print('found valid checkpoint:', ckpt_path)
      args.resume = ckpt_path
      break
    if args.resume is None:
      print('did not find any valid checkpoint to resume from.')

  if args.resume:
    print('resuming from:', args.resume)
    ckpt = torch.load(args.resume, map_location='cpu')
    missing_keys, unexpected_keys = model_without_ddp.load_state_dict(ckpt['model'], strict=False)
    # strict loading but only for params with grad
    assert len(unexpected_keys) == 0, unexpected_keys
    unexpected_keys = set(unexpected_keys)
    for n, p in model_without_ddp.named_parameters():
      if p.requires_grad:
        assert n not in missing_keys
      else:
        assert n in missing_keys, n

    if optimizer is not None:
      optimizer.load_state_dict(ckpt['optimizer'])
    if lr_sched is not None:
      lr_sched.load_state_dict(ckpt['lr_sched'])
    if loss_scaler is not None:
      loss_scaler.load_state_dict(ckpt['loss_scaler'])
    return ckpt['next_epoch']

  elif args.pretrain:
    print('using pretrained model:', args.pretrain)
    ckpt = torch.load(args.pretrain, map_location='cpu')
    ckpt = ckpt['model']
    for n, p in model_without_ddp.named_parameters():
      if not p.requires_grad and n in ckpt:
        del ckpt[n]
    print(model_without_ddp.load_state_dict(ckpt, strict=False))

  return 0

def save_model(args, epoch, model_without_ddp, optimizer, lr_sched, loss_scaler):
  if dist.get_rank() == 0 and ((epoch + 1) % args.save_freq == 0 or (epoch + 1) == args.epochs):
    os.makedirs(args.save_dir, exist_ok=True)
    state_dict = model_without_ddp.state_dict()
    for n, p in model_without_ddp.named_parameters():
      if not p.requires_grad:
        del state_dict[n]
    torch.save({
      'model': state_dict,
      'optimizer': optimizer.state_dict(),
      'lr_sched': lr_sched.state_dict(),
      'loss_scaler': loss_scaler.state_dict(),
      'next_epoch': epoch + 1,
      }, os.path.join(args.save_dir, 'checkpoint-%d.pth' % epoch))

    if args.auto_remove:
      for ckpt in os.listdir(args.save_dir):
        try:
          if not (ckpt.startswith('checkpoint-') and ckpt.endswith('.pth')):
            raise ValueError()
          ckpt_epoch = int(ckpt[len('checkpoint-'):-len('.pth')])
        except ValueError:
          continue

        if ckpt_epoch < epoch:
          ckpt_path = os.path.join(args.save_dir, ckpt)
          print('removing old checkpoint:', ckpt_path)
          os.remove(ckpt_path)
    
