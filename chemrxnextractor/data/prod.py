import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from transformers import AutoTokenizer

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from .utils import InputExample


logger = logging.getLogger(__name__)


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """

    input_ids: List[int]
    attention_mask: List[int]
    token_type_ids: Optional[List[int]] = None
    label_ids: Optional[List[int]] = None
    decoder_mask: Optional[List[bool]] = None


class ProdDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        data_file: str,
        tokenizer: AutoTokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False
    ):
        # Load data features from cache or dataset file
        data_dir = os.path.dirname(data_file)
        fname = os.path.basename(data_file)
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(
                fname,
                tokenizer.__class__.__name__,
                str(max_seq_length)
            ),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_file}")
            examples = read_examples_from_file(data_file)
            self.features = convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=tokenizer.sep_token,
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
            )
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


class PlainProdDataset(Dataset):
    features: List[InputFeatures]
    pad_token_label_id: int = nn.CrossEntropyLoss().ignore_index
    # Use cross entropy ignore_index as padding label id so that only
    # real label ids contribute to the loss later.

    def __init__(
        self,
        data_file: str,
        tokenizer: AutoTokenizer,
        labels: List[str],
        model_type: str,
        max_seq_length: Optional[int] = None,
        overwrite_cache=False,
    ):
        # Load data features from cache or dataset file
        data_dir = os.path.dirname(data_file)
        fname = os.path.basename(data_file)
        cached_features_file = os.path.join(
            data_dir,
            "cached_{}_{}_{}".format(
                fname,
                tokenizer.__class__.__name__,
                str(max_seq_length)
            ),
        )

        if os.path.exists(cached_features_file) and not overwrite_cache:
            logger.info(f"Loading features from cached file {cached_features_file}")
            self.features = torch.load(cached_features_file)
        else:
            logger.info(f"Creating features from dataset file at {data_file}")
            examples = read_examples_from_file(data_file)
            self.features = convert_examples_to_features(
                examples,
                labels,
                max_seq_length,
                tokenizer,
                cls_token=tokenizer.cls_token,
                cls_token_segment_id=0,
                sep_token=tokenizer.sep_token,
                pad_token=tokenizer.pad_token_id,
                pad_token_segment_id=tokenizer.pad_token_type_id,
                pad_token_label_id=self.pad_token_label_id,
            )
            logger.info(f"Saving features into cached file {cached_features_file}")
            torch.save(self.features, cached_features_file)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i) -> InputFeatures:
        return self.features[i]


def read_examples_from_file(file_path) -> List[InputExample]:
    guid_index = 1
    examples = []
    with open(file_path, encoding="utf-8") as f:
        words, labels = [], []
        metainfo = None
        for line in f:
            line = line.rstrip()
            if line.startswith("#\tpassage"):
                metainfo = line
            elif line == "":
                if words:
                    examples.append(InputExample(
                        guid=f"{guid_index}",
                        words=words,
                        metainfo=metainfo,
                        labels=labels
                    ))
                    guid_index += 1
                    words, labels = [], []
            else:
                splits = line.split("\t")
                words.append(splits[0])
                if len(splits) > 1:
                    labels.append(splits[-1])
                else:
                    # Examples could have no label for plain test files
                    labels.append("O")
        if words:
            examples.append(InputExample(
                guid=f"{guid_index}",
                words=words,
                metainfo=metainfo,
                labels=labels
            ))

    return examples


def convert_examples_to_features(
    examples: List[InputExample],
    label_list: List[str],
    max_seq_length: int,
    tokenizer: AutoTokenizer,
    cls_token="[CLS]",
    cls_token_segment_id=0,
    sep_token="[SEP]",
    pad_token=0,
    pad_token_segment_id=0,
    pad_token_label_id=-100,
    sequence_a_segment_id=0,
    sequence_b_segment_id=1,
    mask_padding_with_zero=True,
) -> List[InputFeatures]:
    """ Loads a data file into a list of `InputFeatures`
    """
    label_map = {label: i for i, label in enumerate(label_list)}

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10_000 == 0:
            logger.info("Writing example %d of %d", ex_index, len(examples))

        tokens = []
        label_ids = []
        for word, label in zip(example.words, example.labels):
            word_tokens = tokenizer.tokenize(word)
            # word_tokens = word_tokens[:5]

            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                label_ids.extend([label_map[label]] + [pad_token_label_id] * (len(word_tokens) - 1))

        if len(tokens) > max_seq_length - 2:
            logger.warning("Sequence length exceed {} (cut).".format(max_seq_length))
            tokens = tokens[: (max_seq_length - 2)]
            label_ids = label_ids[: (max_seq_length - 2)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        seq_length = len(input_ids)
        padding_length = max_seq_length - len(input_ids)
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length

        decoder_mask = [(x != pad_token_label_id) for x in label_ids]

        # assert len(input_ids) == max_seq_length
        # assert len(input_mask) == max_seq_length
        # assert len(segment_ids) == max_seq_length
        # assert len(label_ids) == max_seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: {} (length: {})".format(example.guid, seq_length))
            logger.info("tokens: %s", " ".join([str(x) for x in tokens[:seq_length]]))
            logger.info("input_ids: %s", " ".join([str(x) for x in input_ids[:seq_length]]))
            # logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            # logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: %s", " ".join([str(x) for x in label_ids[:seq_length]]))
            logger.info("decode_mask: %s", " ".join([str(x) for x in decoder_mask[:seq_length]]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=input_mask,
                token_type_ids=segment_ids,
                label_ids=label_ids,
                decoder_mask=decoder_mask
            )
        )
    return features


def write_predictions(input_file, output_file, predictions):
    """ Write Product Extraction predictions to file,
        while aligning with the input format.
    """
    with open(output_file, "w") as writer, open(input_file, "r") as f:
        example_id = 0
        for line in f:
            if line.startswith("#\tpassage"):
                writer.write(line)
            elif line == "" or line == "\n":
                writer.write(line)
                if not predictions[example_id]:
                    example_id += 1
            elif predictions[example_id]:
                cols = line.rstrip().split()
                cols.append(predictions[example_id].pop(0))
                writer.write("\t".join(cols) + "\n")
            else:
                logger.warning(
                    "Maximum sequence length exceeded: No prediction for '%s'.", line.split()[0]
                )

