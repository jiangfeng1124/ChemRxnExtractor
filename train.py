""" Trains a product extraction or role labeling model on a dataset.
"""

import warnings
warnings.filterwarnings("ignore")
import argparse
import sys
import os

os.environ["WANDB_DISABLED"] = 'false'

if __name__ == '__main__':
    if len(sys.argv) > 2:
        task = sys.argv[1]
        if task == "prod":
            from chemrxnextractor.prod_args import parse_train_args
            from chemrxnextractor.train import prod_train
            args = parse_train_args(sys.argv[2:])
            prod_train(*args)
        elif task == "role":
            from chemrxnextractor.role_args import parse_train_args
            from chemrxnextractor.train import role_train
            args = parse_train_args(sys.argv[2:])
            role_train(*args)
    else:
        print(f'Usage: {sys.argv[0]} [task] [options]', file=sys.stderr)
