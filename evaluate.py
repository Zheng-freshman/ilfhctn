import argparse
import json
import logging
import os
import random
import warnings

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist

from semilearn.algorithms import get_algorithm, name2alg
from semilearn.core.utils import (
    TBLog,
    count_parameters,
    get_logger,
    get_net_builder,
    get_port,
    over_write_args_from_file,
    send_model_cuda,
)
from semilearn.imb_algorithms import name2imbalg

# Ensure custom algorithms in exp0/exp1 are registered at import time
from exp0 import (
    FractureClassify,
    SemiHealNet,
    SemiFusion,
    SemiFusionCox,
    SemiFusionCoxDual,
    SimMatch,
    SequenceMatch,
    FullySupervised,
    ReFixMatch,
)
from exp1 import (
    NSCLCCoxFusion,
    NSCLCCoxFusionDual,
    NSCLCCoxReFixMatch,
    NSCLCCoxSimMatch,
    NSCLCCoxSequenceMatch,
)


def get_config():
    from semilearn.algorithms.utils import str2bool

    parser = argparse.ArgumentParser(description="Evaluate (USB)")

    # Saving & loading
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("-sn", "--save_name", type=str, default="eval")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--load_path", type=str)
    parser.add_argument("-o", "--overwrite", action="store_true", default=True)
    parser.add_argument("--use_tensorboard", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_aim", action="store_true")

    # Evaluation options
    parser.add_argument("--eval_dest", type=str, default="eval")
    parser.add_argument("--out_key", type=str, default="logits")
    parser.add_argument("--return_logits", type=str2bool, default=False)
    parser.add_argument("--results_out", type=str, default="")

    # Training-related configs kept for algorithm initialization
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--num_train_iter", type=int, default=20)
    parser.add_argument("--num_warmup_iter", type=int, default=0)
    parser.add_argument("--num_eval_iter", type=int, default=10)
    parser.add_argument("--num_log_iter", type=int, default=5)
    parser.add_argument("-nl", "--num_labels", type=int, default=400)
    parser.add_argument("-bsz", "--batch_size", type=int, default=8)
    parser.add_argument("--uratio", type=int, default=1)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--ema_m", type=float, default=0.999)
    parser.add_argument("--ulb_loss_ratio", type=float, default=1.0)

    # Optimizer (unused in pure eval but required by algorithm init)
    parser.add_argument("--optim", type=str, default="SGD")
    parser.add_argument("--lr", type=float, default=3e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--layer_decay", type=float, default=1.0)

    # Backbone
    parser.add_argument("--net", type=str, default="wrn_28_2")
    parser.add_argument("--net_from_name", type=str2bool, default=False)
    parser.add_argument("--use_pretrain", default=False, type=str2bool)
    parser.add_argument("--pretrain_path", default="", type=str)

    # Algorithm configs
    parser.add_argument("-alg", "--algorithm", type=str, default="fixmatch")
    parser.add_argument("--use_cat", type=str2bool, default=True)
    parser.add_argument("--amp", type=str2bool, default=False)
    parser.add_argument("--clip_grad", type=float, default=0)
    parser.add_argument("-imb_alg", "--imb_algorithm", type=str, default=None)

    # Data configs
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("-ds", "--dataset", type=str, default="cifar10")
    parser.add_argument("-nc", "--num_classes", type=int, default=10)
    parser.add_argument("--train_sampler", type=str, default="RandomSampler")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--include_lb_to_ulb", type=str2bool, default="True")

    # CV/NLP/Speech
    parser.add_argument("--img_size", type=int, default=32)
    parser.add_argument("--crop_ratio", type=float, default=0.875)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_length_seconds", type=float, default=4.0)
    parser.add_argument("--sample_rate", type=int, default=16000)

    # Medical fracture dataset
    parser.add_argument("--split_id", type=int, default=0)
    parser.add_argument("--target_type", type=str, default="isMOF")
    parser.add_argument("--fusing", type=bool, default=True)
    parser.add_argument("--freeze_backbone", type=bool, default=True)
    parser.add_argument("--prompt_split", type=bool, default=False)
    parser.add_argument("--pretrained_from", type=str, default="ViT")

    # Distributed/Device
    parser.add_argument("--world-size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("-du", "--dist-url", default="tcp://127.0.0.1:11111", type=str)
    parser.add_argument("--dist-backend", default="nccl", type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--gpu", default=None, type=int)
    parser.add_argument("--multiprocessing-distributed", type=str2bool, default=False)

    # Config file (YAML)
    parser.add_argument("--c", type=str, default="")

    # Add algorithm specific parameters
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    for argument in name2alg[args.algorithm].get_argument():
        parser.add_argument(
            argument.name,
            type=argument.type,
            default=argument.default,
            help=argument.help,
        )

    # Add imbalanced algorithm specific parameters (if any)
    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    if args.imb_algorithm is not None and args.imb_algorithm in name2imbalg:
        for argument in name2imbalg[args.imb_algorithm].get_argument():
            parser.add_argument(
                argument.name,
                type=argument.type,
                default=argument.default,
                help=argument.help,
            )

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)
    args.save_name = args.save_name + "_" + str(args.split_id)
    if hasattr(args, "pretrain_path_vit") and hasattr(args, "pretrain_path_finetune"):
        tmp = args.pretrain_path_vit.split("/")
        tmp[-2] = tmp[-2] + "_" + str(args.split_id)
        args.pretrain_path_vit = "/".join(tmp)
        tmp = args.pretrain_path_finetune.split("/")
        tmp[-2] = tmp[-2] + "_" + str(args.split_id)
        args.pretrain_path_finetune = "/".join(tmp)
    return args


def main(args):
    # Prepare dist url
    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ.get("WORLD_SIZE", 1))

    # Distributed flag
    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    # Device
    if args.gpu == "None":
        args.gpu = None

    # Seed
    assert args.seed is not None
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    # Init process group if in distributed evaluate
    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ.get("RANK", 0))
        dist.init_process_group(
            backend=args.dist_backend,
            init_method=args.dist_url,
            world_size=args.world_size,
            rank=args.rank,
        )

    # Logger
    save_path = os.path.join(args.save_dir, args.save_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)
    logger_level = "INFO"
    tb_log = TBLog(save_path, "tensorboard", use_tensorboard=args.use_tensorboard)
    logger = get_logger(args.save_name, save_path, logger_level)
    logger.info(f"Use GPU: {args.gpu} for evaluate")

    # Constraint args for evaluation
    if args.use_pretrain == True:
        args.use_pretrain = False
        logger.info(f"Disable use_pretrain as it is not supported for evaluation.")

    # Build model
    _net_builder = get_net_builder(args.net, args.net_from_name)
    model = get_algorithm(args, _net_builder, tb_log, logger)
    logger.info(f"Number of Trainable Params: {count_parameters(model.model)}")

    # Device placement
    model.model = send_model_cuda(args, model.model)
    model.ema_model = send_model_cuda(args, model.ema_model, clip_batch=False)
    logger.info(f"Arguments: {model.args}")

    # Load checkpoint if provided
    if args.load_path and os.path.exists(args.load_path):
        try:
            model.load_model(args.load_path)
            logger.info(f"Loaded checkpoint from {args.load_path}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint {args.load_path}: {e}")
    else:
        raise ValueError(f"Checkpoint path {args.load_path} does not exist.")

    # Initialize EMA via hooks (consistent with training flow)
    model.call_hook("before_run")

    # Run evaluation
    eval_dict = model.evaluate(
        eval_dest=args.eval_dest,
        out_key=args.out_key,
        return_logits=args.return_logits,
    )

    # Log results
    for key, item in eval_dict.items():
        logger.info(f"Eval result - {key} : {item}")

    # Optionally save results
    if args.results_out:
        try:
            with open(args.results_out, "w", encoding="utf-8") as f:
                json.dump(eval_dict, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved eval results to {args.results_out}")
        except Exception as e:
            logger.warning(f"Failed to save results to {args.results_out}: {e}")

    # Restore EMA shadow if applied (not strictly necessary outside train)
    # No explicit teardown required for evaluate


if __name__ == "__main__":
    args = get_config()
    port = get_port()
    args.dist_url = "tcp://127.0.0.1:" + str(port)
    main(args)