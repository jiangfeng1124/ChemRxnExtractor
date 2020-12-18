import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union
from transformers import AutoTokenizer
from seqeval.metrics.sequence_labeling import get_entities
from copy import deepcopy
from tqdm.auto import tqdm

import torch
from torch import nn
from torch.utils.data.dataset import Dataset

from .utils import InputExample
from chemrxnextractor.constants import PROD_START_MARKER, PROD_END_MARKER


logger = logging.getLogger(__name__)


@dataclass
class InputFeatures:
    """
    A single set of features of data.
    Property names are the same names as the corresponding inputs to a model.
    """
    input_ids:       List[int]
    attention_mask:  List[int]
    prod_start_mask: List[int]
    prod_end_mask:   List[int]
    prod_mask:       List[int]
    token_type_ids:  Optional[List[int]]  = None
    label_ids:       Optional[List[int]]  = None
    decoder_mask:    Optional[List[bool]] = None


class RoleDataset(Dataset):
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
            examples = self.read_examples_from_file(data_file)
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

    def read_examples_from_file(self, file_path) -> List[InputExample]:
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
                        labels_by_prod = list(zip(*labels))
                        for y in labels_by_prod:
                            assert "B-Prod" in y # make sure there is a Product
                            examples.append(InputExample(
                                guid=f"{guid_index}",
                                metainfo=metainfo,
                                words=words,
                                labels=y
                            ))
                            guid_index += 1
                        words, labels = [], []
                else:
                    cols = line.split("\t")
                    words.append(cols[0])
                    if len(cols) > 1:
                        labels.append(cols[1:])
                    else:
                        labels.append(["O"])

        return examples


class PlainRoleDataset(Dataset):
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
            examples = self.read_examples_from_file(data_file)
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

    def read_examples_from_file(self, file_path) -> List[InputExample]:
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
                        prods = get_entities(labels)
                        for etype, ss, se in prods:
                            # create prod-specific instance
                            assert etype == "Prod"
                            inst_labels = ["O"] * len(words)
                            inst_labels[ss] = "B-Prod"
                            inst_labels[ss+1:se+1] = ["I-Prod"] * (se-ss)
                            examples.append(
                                InputExample(
                                    guid=f"{guid_index}",
                                    words=words,
                                    metainfo=metainfo,
                                    labels=inst_labels
                                )
                            )
                            guid_index += 1
                        words, labels = [], []
                else:
                    cols = line.strip().split('\t')
                    words.append(cols[0])
                    labels.append(cols[1])

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

        prod_start_index = prod_end_index = -1
        for wid, (word, label) in enumerate(
                zip(example.words, example.labels)):
            if label == "B-Prod":
                prod_start_index = len(tokens)
                tokens.append(PROD_START_MARKER)
                label_ids.append(pad_token_label_id)
            elif prod_start_index >= 0 and prod_end_index < 0 and label != "I-Prod":
                prod_end_index = len(tokens)
                tokens.append(PROD_END_MARKER)
                label_ids.append(pad_token_label_id)

            word_tokens = tokenizer.tokenize(word)
            word_tokens = word_tokens[:5] # avoid long chemical names

            if len(word_tokens) > 0:
                tokens.extend(word_tokens)
                # Use the real label id for the first token of the word,
                # and padding ids for the remaining tokens
                # skip unknown labels (used by semi-supervised training with partial annotations
                label_ids.extend(
                    [label_map.get(label, pad_token_label_id)] +
                    [pad_token_label_id] * (len(word_tokens) - 1)
                )

        # Product at the end of sequence
        if prod_start_index >= 0 and prod_end_index < 0:
            prod_end_index = len(tokens)
            tokens.append(PROD_END_MARKER)
            label_ids.append(pad_token_label_id)

        assert prod_start_index >= 0
        assert prod_end_index >= 0

        # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
        if len(tokens) > max_seq_length - 2: # [CLS], [SEP]
            logger.info("Sentence length exceeds max_seq_length: {} ({})"
                        .format(" ".join(tokens), len(tokens)))
            # This will fail if PROD is cut
            tokens = tokens[: (max_seq_length - 2)]
            label_ids = label_ids[: (max_seq_length - 2)]

        tokens += [sep_token]
        label_ids += [pad_token_label_id]
        segment_ids = [sequence_a_segment_id] * len(tokens)

        tokens = [cls_token] + tokens
        label_ids = [pad_token_label_id] + label_ids
        segment_ids = [cls_token_segment_id] + segment_ids

        prod_start_index += 1 # cls_token added to th beginning
        prod_end_index += 1

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        prod_start_mask = [0 for i in range(len(input_ids))]
        prod_start_mask[prod_start_index] = 1
        prod_end_mask = [0 for i in range(len(input_ids))]
        prod_end_mask[prod_end_index] = 1
        prod_mask = [0 for i in range(len(input_ids))]
        prod_mask[prod_start_index:prod_end_index+1] = [1] * (prod_end_index+1-prod_start_index)

        # set segment ids for product
        # segment_ids[prod_start_index:prod_end_index+1] = [1] * (prod_end_index+1-prod_start_index)

        # Zero-pad up to the sequence length.
        seq_length = len(input_ids)
        padding_length = max_seq_length - seq_length
        input_ids += [pad_token] * padding_length
        input_mask += [0 if mask_padding_with_zero else 1] * padding_length
        prod_start_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        prod_end_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        prod_mask += ([0 if mask_padding_with_zero else 1] * padding_length)
        segment_ids += [pad_token_segment_id] * padding_length
        label_ids += [pad_token_label_id] * padding_length
        decoder_mask = [(x != pad_token_label_id) for x in label_ids]

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(prod_start_mask) == max_seq_length
        assert len(prod_end_mask) == max_seq_length
        assert len(prod_mask) == max_seq_length
        assert len(prod_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(label_ids) == max_seq_length

        if ex_index < 1:
            logger.info("*** Example ***")
            logger.info("guid: {} (length: {})".format(example.guid, seq_length))
            logger.info("tokens: " + " ".join([str(x) for x in tokens[:seq_length]]))
            logger.info("input_ids: " + " ".join([str(x) for x in input_ids[:seq_length]]))
            # logger.info("input_mask: %s", " ".join([str(x) for x in input_mask]))
            # logger.info("prod_start_mask: %s", " ".join([str(x) for x in prod_start_mask]))
            # logger.info("prod_end_mask: %s", " ".join([str(x) for x in prod_end_mask]))
            # logger.info("prod_mask: %s", " ".join([str(x) for x in prod_mask]))
            # logger.info("segment_ids: %s", " ".join([str(x) for x in segment_ids]))
            logger.info("label_ids: " + " ".join([str(x) for x in label_ids[:seq_length]]))
            logger.info("decoder_mask: " + " ".join([str(x) for x in decoder_mask[:seq_length]]))

        features.append(
            InputFeatures(
                input_ids=input_ids,
                attention_mask=input_mask,
                prod_start_mask=prod_start_mask,
                prod_end_mask=prod_end_mask,
                prod_mask=prod_mask,
                token_type_ids=segment_ids,
                label_ids=label_ids,
                decoder_mask=decoder_mask
            )
        )
    return features


def write_predictions(input_file, output_file, predictions, align="labeled"):
    """ Write Role predictions to file, while aligning with the input format.
    """
    # align to labeled input file
    if align == "labeled":
        with open(output_file, "w") as writer, open(input_file, "r") as f:
            example_id = 0
            num_rxns = 0
            for line in tqdm(f):
                line = line.rstrip()
                if line.startswith("#\tpassage"):
                    writer.write(line + "\n")
                elif line == "":
                    writer.write("\n")
                    example_id += num_rxns
                else:
                    cols = line.split('\t')
                    num_rxns = len(cols) - 1
                    output_line = [cols[0]]
                    for j, label in enumerate(cols[1:]):
                        output_line.append(label)
                        if label in ["B-Prod", "I-Prod"]:
                            output_line.append(label)
                        else:
                            if predictions[example_id+j]:
                                output_line.append(predictions[example_id+j].pop(0))
                            else:
                                logger.info(
                                    f"Maximum sequence length exceeded: No prediction for {cols[0]}."
                                )

                    writer.write("\t".join(output_line) + "\n")
    # align to plain input file
    elif align == "plain":
        with open(output_file, "w") as writer, open(input_file, "r") as f:
            example_id = 0
            words, labels = [], []
            for line in tqdm(f):
                if line.startswith("#\tpassage"):
                    writer.write(line)
                elif line == "" or line == "\n":
                    if words:
                        prods = get_entities(labels)
                        if len(prods) == 0:
                            writer.write("\n".join(
                                ["\t".join([word, label]) for word, label in zip(words, labels)]
                            ) + "\n\n")
                            words, labels = [], []
                            continue
                        # read len(prods) preds from predictions
                        srl_labels = []
                        for _, ss, se in prods:
                            inst_labels = ["O"] * len(words)
                            inst_labels[ss] = "B-Prod"
                            inst_labels[ss+1:se+1] = ["I-Prod"] * (se-ss)
                            for i, w in enumerate(words):
                                if i < ss or i > se:
                                    if not predictions[example_id]:
                                        logger.info(
                                            f"Maximum sequence length exceeded: No prediction for {w}."
                                        )
                                        continue
                                    inst_labels[i] = predictions[example_id].pop(0)
                            example_id += 1
                            srl_labels.append(inst_labels)
                        assert len(srl_labels) == len(prods)
                        srl_labels_by_token = list(zip(*srl_labels))
                        # writing
                        for i, w in enumerate(words):
                            writer.write(w + "\t" + "\t".join(srl_labels_by_token[i]) + "\n")
                        writer.write("\n")
                    words, labels = [], []
                else:
                    cols = line.strip().split('\t')
                    words.append(cols[0])
                    labels.append(cols[1])

