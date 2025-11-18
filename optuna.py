import argparse
import os
import random
import warnings
import optuna
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from semilearn.algorithms import get_algorithm
from semilearn.core.utils import (
    TBLog,
    count_parameters,
    get_logger,
    get_net_builder,
    send_model_cuda,
    over_write_args_from_file,
    get_port,
)

# ensure algorithm modules are imported and registered
from exp0 import FractureClassify, SemiHealNet, SemiFusion, SemiFusionCox, SemiFusionCoxDual, SimMatch, SequenceMatch, FullySupervised, ReFixMatch
from exp1 import NSCLCCoxFusion, NSCLCCoxFusionDual, NSCLCCoxReFixMatch, NSCLCCoxSimMatch, NSCLCCoxSequenceMatch


def build_args_from_yaml(config_path: str):
    """
    Load args from YAML using the same helper as train.py, and provide sane defaults
    for fields not present in YAML.
    """
    # Create a minimal argparse namespace with defaults
    parser = argparse.ArgumentParser()
    # Common defaults; most training-specific values are expected in YAML
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--save_name", type=str, default="optuna_run")
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_aim", action="store_true")

    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--num_train_iter", type=int, default=20)
    parser.add_argument("--num_warmup_iter", type=int, default=0)
    parser.add_argument("--num_eval_iter", type=int, default=10)
    parser.add_argument("--num_log_iter", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--uratio", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--ema_m", type=float, default=0.999)
    parser.add_argument("--ulb_loss_ratio", type=float, default=1.0)

    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--layer_decay", type=float, default=1.0)

    parser.add_argument("--net", type=str, default="wrn_28_2")
    parser.add_argument("--net_from_name", type=bool, default=False)
    parser.add_argument("--use_pretrain", type=bool, default=False)
    parser.add_argument("--pretrain_path", type=str, default="")

    parser.add_argument("--algorithm", type=str, default="softmatch_fusion_cox_dual")
    parser.add_argument("--use_cat", type=bool, default=True)
    parser.add_argument("--amp", type=bool, default=False)
    parser.add_argument("--clip_grad", type=float, default=0)

    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--dataset", type=str, default="cifar10")
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--include_lb_to_ulb", type=bool, default=True)

    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--target_type", type=str, default="isMOF")
    parser.add_argument("--fusing", type=bool, default=True)
    parser.add_argument("--freeze_backbone", type=bool, default=True)
    parser.add_argument("--prompt_split", type=bool, default=False)
    parser.add_argument("--pretrained_from", type=str, default="ViT")

    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--world_size", type=int, default=1)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--dist_url", type=str, default="tcp://127.0.0.1:11111")
    parser.add_argument("--dist_backend", type=str, default="nccl")
    parser.add_argument("--multiprocessing_distributed", type=bool, default=False)

    parser.add_argument("--c", type=str, default="")
    args = parser.parse_args([])
    args.c = config_path
    # overlay YAML values into args
    over_write_args_from_file(args, args.c)
    # normalize None as string
    if args.gpu == "None":
        args.gpu = None
    # save name suffix for split
    args.save_name = args.save_name + "_" + str(args.split_id)
    if hasattr(args, 'pretrain_path_vit') and hasattr(args, 'pretrain_path_finetune'):
        tmp = args.pretrain_path_vit.split("/")
        tmp[-2] = tmp[-2] + '_' + str(args.split_id)
        args.pretrain_path_vit = '/'.join(tmp)
        tmp = args.pretrain_path_finetune.split("/")
        tmp[-2] = tmp[-2] + '_' + str(args.split_id)
        args.pretrain_path_finetune = '/'.join(tmp)
    return args


def run_training_once(args):
    """Run a single training session and return a scalar metric to optimize."""

    # init seeds
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # distributed flags are simplified for Optuna
    args.distributed = False
    ngpus_per_node = torch.cuda.device_count()

    # logger and tensorboard
    save_path = os.path.join(args.save_dir, args.save_name)
    if os.path.exists(save_path) and args.overwrite and args.resume is False:
        import shutil
        shutil.rmtree(save_path)
    tb_log = TBLog(save_path, "tensorboard", use_tensorboard=args.use_tensorboard)
    logger = get_logger(args.save_name, save_path, "INFO")
    logger.info(f"Use GPU: {args.gpu} for training")

    # build model
    _net_builder = get_net_builder(args.net, args.net_from_name)
    model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f"Number of Trainable Params: {count_parameters(model.model)}")
    # devices
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    # resume support
    if args.resume and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
        except Exception:
            logger.info("Fail to resume load path {}".format(args.load_path))
            args.resume = False
    else:
        logger.info("Resume load path {} does not exist".format(args.load_path))

    # optional warmup/finetune
    if hasattr(model, "warmup"):
        logger.info(("Warmup stage"))
        model.warmup()

    # train
    logger.info("Model training")
    model.train()

    # finetune when available
    if hasattr(model, "finetune"):
        logger.info("Finetune stage")
        model.finetune()

    # collect metric
    metric = None
    # prefer dual-branch metrics when available
    if hasattr(model, 'best_eval_acc_list'):
        metric = sum(model.best_eval_acc_list)
    else:
        metric = model.best_eval_acc
    return metric


def _is_excluded_param(name: str, value) -> bool:
    name_l = name.lower()
    # exclude by name patterns
    if 'path' in name_l or 'url' in name_l:
        return True
    # exclude specific names
    if name in {'save_name', 'split_id', 'net', 'algorithm', 'world_size'}:
        return True
    # exclude booleans
    if isinstance(value, bool):
        return True
    return False


def _suggest_param(trial: optuna.Trial, name: str, value, args_cli):
    """Heuristic search space for known keys; fallback for generic numeric types."""
    # manual spaces for common training hyperparams
    if name == 'lr':
        return trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    if name == 'weight_decay':
        return trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)
    if name == 'momentum':
        return trial.suggest_float('momentum', 0.8, 0.99)
    if name == 'ema_m':
        return trial.suggest_float('ema_m', 0.9, 0.9999)
    if name == 'layer_decay':
        return trial.suggest_float('layer_decay', 0.5, 1.0)
    if name == 'ulb_loss_ratio':
        return trial.suggest_float('ulb_loss_ratio', 0.5, 2.0)
    if name == 'crop_ratio':
        return trial.suggest_float('crop_ratio', 0.7, 0.95)
    if name == 'batch_size':
        base = int(value) if isinstance(value, int) else 8
        low = max(4, base // 2)
        high = max(base * 2, 32)
        return trial.suggest_int('batch_size', low, high, step=4)
    if name == 'num_train_iter':
        base = int(value) if isinstance(value, int) else 200
        low = max(20, base)
        high = max(base * 5, 500)
        return trial.suggest_int('num_train_iter', low, high, step=10)
    if name == 'num_warmup_iter':
        nti = int(getattr(args_cli, 'num_train_iter', 0) or (int(value) if isinstance(value, int) else 200))
        max_warmup = max(10, int(nti * 0.3))
        return trial.suggest_int('num_warmup_iter', 0, max_warmup, step=5)
    if name == 'num_eval_iter':
        nti = int(getattr(args_cli, 'num_train_iter', 0) or (int(value) if isinstance(value, int) else 200))
        return trial.suggest_int('num_eval_iter', 10, max(10, nti // 5), step=5)
    if name == 'num_log_iter':
        nti = int(getattr(args_cli, 'num_train_iter', 0) or (int(value) if isinstance(value, int) else 200))
        return trial.suggest_int('num_log_iter', 5, max(5, nti // 10), step=5)
    if name == 'uratio':
        return trial.suggest_int('uratio', 1, 4)
    if name == 'num_workers':
        return trial.suggest_int('num_workers', 1, 4)
    if name == 'optim':
        return trial.suggest_categorical('optim', ['SGD', 'AdamW'])
    if name == 'surv_loss':
        return trial.suggest_categorical('surv_loss', ['ce', 'nll', 'cox'])

    # fallback spaces: numeric only
    if isinstance(value, float):
        low = 0.0
        high = max(1e-8, float(value) * 2.0) if value != 0 else 1.0
        return trial.suggest_float(name, low, high)
    if isinstance(value, int):
        low = max(1, int(value) // 2)
        high = max(int(value) * 2, int(value) + 1)
        return trial.suggest_int(name, low, high)
    # otherwise do not tune
    return value


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter tuning for time noise intensities")
    parser.add_argument("--c", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--algorithm", type=str, default="softmatch_fusion_cox_dual")
    parser.add_argument("--study_name", type=str, default="t_intensity_tuning")
    parser.add_argument("--storage", type=str, default=None, help="Optuna storage URL, e.g., sqlite:///optuna.db")
    parser.add_argument("--n_trials", type=int, default=20)
    parser.add_argument("--direction", type=str, default="maximize")
    parser.add_argument("--weak_min", type=float, default=0.0)
    parser.add_argument("--weak_max", type=float, default=0.5)
    parser.add_argument("--strong_min", type=float, default=0.5)
    parser.add_argument("--strong_max", type=float, default=2.0)
    parser.add_argument("--weak_step", type=float, default=0.05)
    parser.add_argument("--strong_step", type=float, default=0.05)
    args_cli = parser.parse_args()

    def objective(trial: optuna.Trial):
        # load and overlay YAML
        args = build_args_from_yaml(args_cli.c)
        # set algorithm explicitly
        args.algorithm = args_cli.algorithm
        # inject suggested hyperparameters
        t_weak = trial.suggest_float("t_weak_intensity", args_cli.weak_min, args_cli.weak_max, step=args_cli.weak_step)
        t_strong = trial.suggest_float("t_strong_intensity", args_cli.strong_min, args_cli.strong_max, step=args_cli.strong_step)
        setattr(args, 't_weak_intensity', t_weak)
        setattr(args, 't_strong_intensity', t_strong)
        # iterate over args to auto-tune permitted parameters
        for k, v in list(vars(args).items()):
            if _is_excluded_param(k, v):
                continue
            # keep user-provided CLI overrides fixed
            if hasattr(args_cli, k):
                continue
            try:
                suggested = _suggest_param(trial, k, v, args_cli)
                setattr(args, k, suggested)
            except Exception:
                # if suggestion fails, keep original
                pass
        epoch = args.num_train_iter*args.epoch//330 # 330 is the number of case per epoch
        # ensure overwrite between trials
        if not hasattr(args, 'overwrite'):
            args.overwrite = True
        # give each trial a unique save_name
        args.save_name = f"{args.save_name}_trial_{trial.number}"
        # run training and get metric
        metric = run_training_once(args)
        return metric

    study = optuna.create_study(study_name=args_cli.study_name, storage=args_cli.storage, direction=args_cli.direction)
    study.optimize(objective, n_trials=args_cli.n_trials)

    print("Best trial:")
    print(f"  Value: {study.best_trial.value}")
    print("  Params:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")


if __name__ == "__main__":
    # randomize port to avoid conflicts in potential distributed setups
    main()