import argparse
import os

import torch.cuda

from src.pretraining.simclr_trainer import SimClrTrainer
from src.pretraining.xclr_trainer import XClrTrainer


def get_args():
    parser = argparse.ArgumentParser()
    # Required positional arguments
    parser.add_argument('alg', choices=['simclr', 'xclr'], help='Training algorithm to use')
    parser.add_argument('dataset_path', type=str, help='Dataset to load for training')

    # Optional arguments with defaults
    parser.add_argument('--batch_size', '-b', required=True, type=int, help='Batch size for training')
    parser.add_argument(
        '--device',
        '-d',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str,
        help='Device to use (cpu/cuda)'
    )
    parser.add_argument('--label_range', '-lr', default=50, type=int, help='Range of labels')
    parser.add_argument('--head_out_features', '-hof', default=128, type=int, help='Output features for the head layer')
    parser.add_argument('--tau', '-t', default=0.1, type=float, help='Temperature parameter for contrastive loss')
    parser.add_argument('--tau_s', '-ts', default=0.1, type=float, help='Secondary temperature parameter')
    parser.add_argument('--num_workers', '-nw', default=8, type=int, help='Number of DataLoader workers')
    parser.add_argument('--epochs', '-e', default=100, type=int, help='Number of epochs to train for')
    parser.add_argument(
        '--encoder_load_path',
        '-elp',
        default=None,
        type=str,
        help='Path to load pretrained encoder (if any)'
    )

    return parser.parse_args()


def create_dir_if_not_exist(path: str):
    if not os.path.exists(path):
        os.makedirs(path)

if __name__ == '__main__':
    args = get_args()

    shared_trainer_args = {
        "dataset_path": args.dataset_path,
        "batch_size": args.batch_size,
        "device": args.device,
        "head_out_features": args.head_out_features,
        "tau": args.tau,
        "num_workers_dl": args.num_workers,
        "epochs": args.epochs,
        "encoder_load_path": args.encoder_load_path
    }

    base_checkpoint_path = 'checkpoints/encoders/'
    cp_dir = os.path.join(base_checkpoint_path, args.alg)
    create_dir_if_not_exist(cp_dir)

    if args.alg == 'simclr':
        trainer = SimClrTrainer(
            **shared_trainer_args,
            encoder_checkpoint_base_path=cp_dir
        )
    else:
        trainer = XClrTrainer(
            **shared_trainer_args,
            label_range=args.label_range,
            tau_s=args.tau_s,
            encoder_checkpoint_base_path=cp_dir,
        )

    trainer.train()
