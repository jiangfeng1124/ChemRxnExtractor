""" Loads a trained product extraction or role labeling model and
    makes predictions on a plain dataset.
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
            from chemrxnextractor.prod_args import parse_predict_args
            from chemrxnextractor.train import prod_predict
            args = parse_predict_args(sys.argv[2:])
            prod_predict(*args)
        elif task == "role":
            from chemrxnextractor.role_args import parse_predict_args
            from chemrxnextractor.train import role_predict
            args = parse_predict_args(sys.argv[2:])
            role_predict(*args)
    else:
        print(f'Usage: {sys.argv[0]} [task] [options]', file=sys.stderr)
