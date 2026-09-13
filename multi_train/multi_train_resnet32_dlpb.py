import os
import sys
import argparse

from pathlib import Path
from traceback import print_exc

DLPB_DIR = Path(__file__).resolve().parents[1]
if str(DLPB_DIR) not in sys.path:
    sys.path.insert(0, str(DLPB_DIR))
if str(DLPB_DIR.parent) not in sys.path:
    sys.path.append(str(DLPB_DIR.parent))

from train.train_resnet32_dlpb import run

from setup import clear
from args import get_args_parser


setting_dict = dict(
    cifar100_resnet32_dlpb='data --dataset_type CIFAR100 --train-size 32 32 --test-size 32 32 --train-resize-mode RandomCrop --random-crop-pad 4 --mean 0.5071 0.4865 0.4409 --std 0.2673 0.2564 0.2762 --epoch 300 --GA_lamb -0.5 --alpha 1 --optimizer sgd --nesterov --lr 0.1 --weight-decay 5e-4 --scheduler multistep --milestones 150 225 -b 128 -j 8',
)


def get_multi_args_parser():
    parser = argparse.ArgumentParser(description='pytorch-cifar-examples', add_help=True, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('setup', type=str, nargs='+', choices=setting_dict.keys(), help='experiment setup')
    parser.add_argument('-m', '--model-name', type=str, nargs='+', default=['ga_resnet32_gram_dim_32_branch_3'], help='list of model names')
    parser.add_argument('-c', '--cuda', type=str, default='0', help='cuda device')
    parser.add_argument('-o', '--output-dir', type=str, default='log', help='log dir')
    parser.add_argument('-p', '--project-name', type=str, default='pytorch-cifar 32 ablation', help='project name used for wandb logger')
    parser.add_argument('-w', '--who', type=str, default='dongu', help='enter your name')
    parser.add_argument('--use-wandb', action='store_true', default=False, help='use wandb')
    parser.add_argument('-exp', '--exp-name', type=str, default=None, help='experiment name for each run')
    return parser


def pass_required_variable_from_previous_args(args, prev_args=None):
    if prev_args:
        required_vars = ['gpu', 'world_size', 'distributed', 'is_rank_zero', 'device']
        for var in required_vars:
            exec(f"args.{var} = prev_args.{var}")


def save_arguments(args, is_master):
    if is_master:
        print("Multiple Train Setting")
        print(f" - model (num={len(args.model_name)}): {', '.join(args.model_name)}")
        print(f" - setting (num={len(args.setup)}): {', '.join(args.setup)}")
        print(f" - cuda: {args.cuda}")
        print(f" - output dir: {args.output_dir}")

        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        with open(os.path.join(args.output_dir, 'last_multi_args.txt'), 'wt') as f:
            f.write(" ".join(sys.argv))


if __name__ == '__main__':
    is_master = os.environ.get('LOCAL_RANK', None) is None or int(os.environ['LOCAL_RANK']) == 0
    multi_args_parser = get_multi_args_parser()
    multi_args = multi_args_parser.parse_args()
    save_arguments(multi_args, is_master)
    prev_args = None

    for setup in multi_args.setup:
        args_parser = get_args_parser()
        args = args_parser.parse_args(setting_dict[setup].split(' '))
        pass_required_variable_from_previous_args(args, prev_args)
        for model_name in multi_args.model_name:
            args.setup = setup
            args.model_name = model_name
            args.seed = 42
            for option_name in ['cuda', 'output_dir', 'project_name', 'who', 'use_wandb', 'exp_name']:
                exec(f"args.{option_name} = multi_args.{option_name}")
            try:
                run(args)
            except:
                print(print_exc())
            clear(args)
        prev_args = args