from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import os

from transformers import HfArgumentParser
from transformers import TrainingArguments


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier."}
    )
    use_fast: bool = field(
        default=False, metadata={"help": "Set this flag to use fast tokenization."}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    use_crf: bool = field(
        default=False, metadata={"help": "Whether using CRF for inference."}
    )
    use_cls: bool = field(
        default=False, metadata={"help": "Whether concatenating token representation with [CLS]."}
    )


@dataclass
class ExTrainingArguments(TrainingArguments):
    crf_learning_rate: float = field(
        default=5e-3, metadata={"help": "The initial learning rate of CRF parameters for Adam."}
    )


@dataclass
class DataArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    data_dir: str = field(
        metadata={"help": "The input data dir. Should contain the .txt files for a CoNLL-2003-formatted task."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels. If not specified, CoNLL-2003 labels are used."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


@dataclass
class PredictArguments:
    input_file: str = field(
        metadata={"help": "Path to a file containing sentences to be extracted (can be a single column file without labels)."}
    )
    output_file: str = field(
        metadata={"help": "Path to a file saving the outputs."}
    )
    labels: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a file containing all labels."},
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer "
                  "than this will be truncated, sequences shorter will be padded."},
    )
    batch_size: int = field(
        default=8, metadata={"help": "Batch size for prediction."}
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached test data."}
    )
    no_cuda: bool = field(
        default=False,
        metadata={"help": "Do not use CUDA even when it is available."}
    )


def parse_train_args(args):
    parser = HfArgumentParser((ModelArguments, DataArguments, ExTrainingArguments))
    if len(args) == 1 and args[0].endswith(".json"):
        model_args, data_args, train_args = parser.parse_json_file(
            json_file=os.path.abspath(args[0]))
    else:
        model_args, data_args, train_args = parser.parse_args_into_dataclasses(args=args)

    return model_args, data_args, train_args


def parse_predict_args(args):
    parser = HfArgumentParser((ModelArguments, PredictArguments))
    if len(args) == 1 and args[0].endswith(".json"):
        model_args, predict_args = parser.parse_json_file(
            json_file=os.path.abspath(args[0]))
    else:
        model_args, predict_args = parser.parse_args_into_dataclasses(args=args)

    return model_args, predict_args

