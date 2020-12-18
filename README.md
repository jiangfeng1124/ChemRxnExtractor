# Chemical Reaction Extraction from Scientific Literature

This repository contains code/data for extracting chemical reactions from scientific literature.

## Installation

### Pre-requirements

1. pytorch (>=1.5.0)
2. transformers (>=3.0.2)

### Install from source
1. `git clone https://github.com/jiangfeng1124/ChemRxnExtractor`
2. `cd ChemRxnExtractor`
3. `pip install -e .`

## Data

### Product Extraction

### Reaction Role Extraction


## Pre-trained Model: ChemBERT

Our model is greatly benefited from a pre-trained model named ChemBERT.
To train a new model on your own datasets, download ChemBERT from XXX.

## Usage
Using this extractor is straightforward:
```
from chemrxnextractor import RxnExtractor

model_dir="models" # directory saving both prod and role models
rxn_extractor = RxnExtractor(model_dir)

# test_file contains texts line by line
with open(test_file, "r") as f:
    sents = f.read().splitlines()
rxns = rxn_extractor.get_reactions(sents)
```

See `pipeline.py` for more details.
GPU is used as the default device, please ensure that you have at least >5G allocatable GPU memory.

## Train and Evaluation

It is also easy to train a new model (either product or role extraction) using your own data.

To train a product extraction model, run:
```
python train.py <task> <config_path>|<options>
```
where `<task>` is either "prod" or "role" depending on the task of interest, `<config_path>` is a json file containing required hyper-parameters, and `<options>` are instead explicitly-specified hyper-parameters.

For example:
```
python train.py prod configs/prod_train.json
```

## Predict

To generate predictions for raw inputs, run:
```
python predict.py <task> <config_json>
```

For example:
```
python predict.py prod configs/prod_predict.json
```

## Results

