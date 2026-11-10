from contextlib import contextmanager
import gc
import time
import math
from pathlib import Path
import os

import torch
import deepspeed.comm.comm as dist
import imageio
from safetensors import safe_open


DTYPE_MAP = {
    'float32': torch.float32,
    'float16': torch.float16,
    'bfloat16': torch.bfloat16,
    'float8': torch.float8_e4m3fn,
    'float8_e4m3fn': torch.float8_e4m3fn,
    'float8_e5m2': torch.float8_e5m2,
}
VIDEO_EXTENSIONS = set()
for x in imageio.config.video_extensions:
    VIDEO_EXTENSIONS.add(x.extension)
    VIDEO_EXTENSIONS.add(x.extension.upper())
AUTOCAST_DTYPE = None


def get_rank():
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


@contextmanager
def zero_first():
    if not is_main_process():
        dist.barrier()
    yield
    if is_main_process():
        dist.barrier()


@contextmanager
def one_at_a_time():
    for i in range(int(os.environ['LOCAL_SIZE'])):
        if i == int(os.environ['LOCAL_RANK']):
            yield
        dist.barrier()


def empty_cuda_cache():
    gc.collect()
    torch.cuda.empty_cache()


@contextmanager
def log_duration(name):
    start = time.time()
    try:
        yield
    finally:
        print(f'{name}: {time.time()-start:.3f}')


def load_safetensors(path):
    tensors = {}
    with safe_open(path, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
    return tensors


def load_state_dict(path):
    path = str(path)
    if path.endswith('.safetensors'):
        sd = load_safetensors(path)
    else:
        sd = torch.load(path, weights_only=True)
    for key in sd:
        if key.endswith('scale_input') or key.endswith('scale_weight'):
            raise ValueError('fp8_scaled weights are not supported. Please use bf16 or normal fp8 weights.')
    return sd


def iterate_safetensors(path):
    path = Path(path)
    if path.is_dir():
        safetensors_files = list(path.glob('*.safetensors'))
        if len(safetensors_files) == 0:
            raise FileNotFoundError(f'Cound not find safetensors files in directory {path}')
    else:
        if path.suffix != '.safetensors':
            raise ValueError(f'Expected {path} to be a safetensors file')
        safetensors_files = [path]
    for filename in safetensors_files:
        with safe_open(str(filename), framework="pt", device="cpu") as f:
            for key in f.keys():
                if key.endswith('scale_input') or key.endswith('scale_weight'):
                    raise ValueError('fp8_scaled weights are not supported. Please use bf16 or normal fp8 weights.')
                yield key, f.get_tensor(key)


def round_to_nearest_multiple(x, multiple):
    return int(round(x / multiple) * multiple)


def round_down_to_multiple(x, multiple):
    return int((x // multiple) * multiple)


def time_shift(mu: float, sigma: float, t: torch.Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15):
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_t_distribution(model_config):
    timestep_sample_method = model_config.get('timestep_sample_method', 'logit_normal')

    if timestep_sample_method == 'logit_normal':
        dist = torch.distributions.normal.Normal(0, 1)
    elif timestep_sample_method == 'uniform':
        dist = torch.distributions.uniform.Uniform(0, 1)
    else:
        raise NotImplementedError()

    n_buckets = 10_000
    delta = 1 / n_buckets
    min_quantile = delta
    max_quantile = 1 - delta
    quantiles = torch.linspace(min_quantile, max_quantile, n_buckets)
    t = dist.icdf(quantiles)

    if timestep_sample_method == 'logit_normal':
        sigmoid_scale = model_config.get('sigmoid_scale', 1.0)
        t = t * sigmoid_scale
        t = torch.sigmoid(t)

    return t


def slice_t_distribution(t, min_t=0.0, max_t=1.0):
    start = torch.searchsorted(t, min_t).item()
    end = torch.searchsorted(t, max_t).item()
    return t[start:end]


def sample_t(t, batch_size, quantile=None):
    if quantile is not None:
        i = (torch.full((batch_size,), quantile) * len(t)).to(torch.int32)
    else:
        i = torch.randint(0, len(t), size=(batch_size,))
    return t[i]

def linear_scheduler_steps(config, steps_per_epoch):
    """Return (end_factor, decay_steps) for the linear schedule.

    Without max_steps or lr_scheduler_end_lr, decay_steps stays epochs * steps_per_epoch.
    max_steps caps the run. With an end learning rate, decay starts after warmup and
    reaches that floor on the last step of the capped run.
    """
    budget = config['epochs'] * steps_per_epoch
    capped = 'max_steps' in config
    if capped:
        budget = min(budget, config['max_steps'])
    end_factor = 0.0
    decay_steps = budget
    if 'lr_scheduler_end_lr' in config:
        initial_lr = config['optimizer']['lr']
        end_lr = config['lr_scheduler_end_lr']
        if not 0 <= end_lr <= initial_lr or initial_lr <= 0:
            raise ValueError('lr_scheduler_end_lr must be between 0 and optimizer.lr, with optimizer.lr > 0')
        end_factor = end_lr / initial_lr
        decay_steps = budget - config.get('warmup_steps', 0)
        if decay_steps <= 0:
            raise ValueError('warmup_steps must be less than the total training steps')
    elif capped:
        decay_steps = budget - config.get('warmup_steps', 0)
        if decay_steps <= 0:
            raise ValueError('warmup_steps must be less than max_steps')
    return end_factor, decay_steps


def apply_param_group_lr_scales(param_groups, base_lr):
    """Turn optional lr_scale into an absolute lr. Groups without it are unchanged."""
    resolved = []
    for group in param_groups:
        if not isinstance(group, dict):
            resolved.append(group)
            continue
        group = dict(group)
        if 'lr_scale' in group:
            if 'lr' in group:
                raise ValueError('a parameter group cannot set both lr and lr_scale')
            if base_lr is None:
                raise ValueError('lr_scale requires the optimizer learning rate')
            group['lr'] = base_lr * group.pop('lr_scale')
        resolved.append(group)
    return resolved
