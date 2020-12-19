import logging
import argparse
import json

import warnings
warnings.filterwarnings("ignore")
from chemrxnextractor import RxnExtractor

if __name__ == '__main__':
    logging.basicConfig(format="%(message)s", level=logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, default="output.json")
    args = parser.parse_args()

    rxn_extractor = RxnExtractor(model_dir=args.model_dir)

    with open(args.input, "r") as f:
        sents = f.read().splitlines()
    rxns = rxn_extractor.get_reactions(sents)

    with open(args.output, "w") as writer:
        json.dump(rxns, writer)

