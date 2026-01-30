import numpy as np
import random
import os
import yaml
import json
import logging
import datetime
import torch.distributed as dist
from tools.optimization import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

DIR_PATH = os.path.dirname(os.path.realpath(__file__))
import os
import torch
import torch.distributed as dist
import wandb


def init_distributed(config):
    if not torch.cuda.is_available():
        device = torch.device("cpu")
        return False, device, 0, 1
    env_world = os.environ.get("WORLD_SIZE", None)
    env_rank = os.environ.get("RANK", None)
    env_local_rank = os.environ.get("LOCAL_RANK", None)
    if env_world is not None:
        world_size = int(env_world)
        rank = int(env_rank) if env_rank is not None else 0
        local_rank = int(env_local_rank) if env_local_rank is not None else rank
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
        )
        return True, device, rank, world_size
    gpu_index = getattr(config, "gpu_index", 0)
    device = torch.device(f"cuda:{gpu_index}")
    return False, device, 0, 1


def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_args(filename, args):
    with open(filename, 'r') as stream:
        data_loaded = yaml.safe_load(stream)
    for key, group in data_loaded.items():
        for key, val in group.items():
            setattr(args, key, val)


def write_json(filename, content):
    with open(filename, 'w') as f:
        json.dump(content, f)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_optimizer(model, config, logger):
    regular_params, special_params = [], []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'peft_tuner' in name:
                special_params.append(param)
            else:
                regular_params.append(param)
    param_groups = [
        {'params': regular_params, 'lr': config.lr},
        {'params': special_params, 'lr': config.peft_lr}
    ]
    ft_layer = config.FT_Layer if hasattr(config, 'FT_Layer') else None
    if ft_layer:
        ft_params= []
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            for layer in ft_layer:
                if layer in name:
                    param.requires_grad = True
                    ft_params.append(param)
        param_groups = [
            {'params': ft_params, 'lr': config.lr},
        ]
        print("========== Trainable parameters ==========")
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"{name:60s} {str(param.shape):>20s} {param.numel():>10d}")
        print("==========================================")

    if config.optimizer == 'Adam':
        optimizer = torch.optim.Adam(param_groups, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = torch.optim.SGD(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    elif config.optimizer == 'AdamW':
        optimizer = torch.optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay, betas=(0.9, 0.999))
    else:
        raise ValueError(f"Unsupported optimizer: {config.optimizer}")
    logger.info(f"Using Optimizer: {config.optimizer}, base learning rate: {config.lr}")

    return optimizer


def get_scheduler(optimizer, config, num_batches=-1):
    if config.scheduler == 'StepLR':
        if hasattr(config, 'train_trigger') and config.train_trigger == 'step':
            step_size_in_steps = config.step_size
        else:
            step_size_in_steps = config.step_size * num_batches
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size_in_steps, gamma=config.gamma)
    elif config.scheduler == 'linear_w_warmup' or config.scheduler == 'cosine_w_warmup':
        assert num_batches != -1
        num_training_steps = num_batches * config.epochs
        num_warmup_steps = int(config.warmup_proportion * num_training_steps)
        if config.scheduler == 'linear_w_warmup':
            scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
        if config.scheduler == 'cosine_w_warmup':
            scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)
    return scheduler


def step_scheduler(scheduler, config, bid, num_batches):
    if config.scheduler in ['StepLR']:
        if bid + 1 == num_batches:    # end of the epoch
            scheduler.step()
    elif config.scheduler in ['linear_w_warmup', 'cosine_w_warmup']:
        scheduler.step()

    return scheduler


def load_model(model, pretrain_path=None):
    if pretrain_path is None:
        print("\033[93mWarning: No pretrain path provided, skipping loading.\033[0m")
        return model
    pth_dict = torch.load(pretrain_path, map_location=lambda storage, loc: storage)
    model_dict = model.state_dict()
    matched_keys = []
    mismatched_keys = []
    for key in pth_dict.keys():
        if key in model_dict:
            if pth_dict[key].shape == model_dict[key].shape:
                matched_keys.append(key)
            else:
                mismatched_keys.append((key, f"Shape mismatch: {pth_dict[key].shape} vs {model_dict[key].shape}"))
        else:
            mismatched_keys.append((key, "Key not found in model"))
    if mismatched_keys:
        print("\033[91mWarning: Found {} mismatched keys:\033[0m".format(len(mismatched_keys)))
        for key, reason in mismatched_keys:
            print("\033[91m  - {}: {}\033[0m".format(key, reason))
    if matched_keys:
        filtered_pth_dict = {k: v for k, v in pth_dict.items() if k in matched_keys}
        model_dict.update(filtered_pth_dict)
        model.load_state_dict(model_dict)
        print("\033[92mLoaded {} parameters from {}.\033[0m".format(len(matched_keys), pretrain_path))
    else:
        print("\033[93mNo parameters loaded from pretrained model.\033[0m")
    return model

def print_model_parameters(model, logger):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    non_trainable_params = total_params - trainable_params
    model_size_mb = total_params * 4 / (1024 ** 2)
    logger.info(f"Total parameters: {total_params:,}  |  Model size: {model_size_mb:.2f}Mb")
    logger.info(f"Trainable parameters: {trainable_params:,}  |  Non-trainable parameters: {non_trainable_params:,}")


def setup_logger(save_dir: str, log_name: str = None):
    os.makedirs(save_dir, exist_ok=True)
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.handlers:
        logger.handlers.clear()
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    if log_name is None or not log_name.strip():
        log_name = f"run-{ts}.log"
    if not log_name.endswith(".log"):
        log_name += ".log"
    log_path = os.path.join(save_dir, log_name)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(ch)
    logger.info(f"Logging to {log_path}")
    return logger, log_path


def set_loss_buffers(loss_buffers, loss_out):
    first_key = next(iter(loss_out.keys()))
    loss_total = loss_out[first_key]
    # write all keys to buffer (support tensors or numbers)
    for k, v in loss_out.items():
        if isinstance(v, torch.Tensor):
            loss_buffers[k].append(v.detach().cpu().item())
        else:
            loss_buffers[k].append(float(v))
    return loss_buffers, loss_total


def set_loss_postfix(loss_buffers, postfix):
    for key, arr in loss_buffers.items():
        if len(arr) > 0:
            postfix[f"{key}"] = np.mean(arr[-50:])
    return postfix


def set_wandb_log(loss_buffers, optimizer, epoch, global_step):
    log_dict = {}
    for key, arr in loss_buffers.items():
        if len(arr) > 0:
            log_dict[key] = arr[-1]
    log_dict["lr"] = optimizer.param_groups[0]["lr"]
    log_dict["epoch"] = epoch + 1
    log_dict["global_step"] = global_step
    wandb.log(log_dict, step=global_step)
