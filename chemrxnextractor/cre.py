import logging
import os
import sys
from seqeval.metrics.sequence_labeling import get_entities
from collections import defaultdict
from dataclasses import dataclass
from tqdm.auto import tqdm

import chemrxnextractor as cre
from chemrxnextractor.models import BertForTagging
from chemrxnextractor.models import BertCRFForTagging
from chemrxnextractor.models import BertForRoleLabeling
from chemrxnextractor.models import BertCRFForRoleLabeling
from chemrxnextractor.data.utils import InputExample
from .utils import create_logger

import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers import AutoConfig, AutoTokenizer
from transformers.data.data_collator import default_data_collator


class RxnDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class RxnExtractor(object):
    def __init__(self, model_dir, batch_size=64, use_cuda=True):
        self.model_dir = model_dir
        self.batch_size = batch_size
        self.pad_token_label_id = nn.CrossEntropyLoss().ignore_index

        self.prod_max_seq_len = 256
        self.role_max_seq_len = 512

        self.device = torch.device(
            "cuda"
            if (use_cuda and torch.cuda.is_available())
            else "cpu"
        )

        self.load_model()

    def load_model(self):
        prod_model_dir = os.path.join(self.model_dir, "prod")
        if os.path.isdir(prod_model_dir):
            sys.stderr.write(f"Loading product extractor from {prod_model_dir}...")
            config = AutoConfig.from_pretrained(prod_model_dir)
            prod_tokenizer = AutoTokenizer.from_pretrained(
                prod_model_dir,
                use_fast=True
            )
            model_class = (
                BertCRFForTagging
                if "BertCRFForTagging" in config.architectures
                else BertForTagging
            )
            prod_extractor = model_class.from_pretrained(
                prod_model_dir,
                config=config
            )
            prod_labels = ["O"] * len(config.id2label)
            for i, label in config.id2label.items():
                prod_labels[int(i)] = label
            sys.stderr.write("done\n")
        else:
            sys.stderr.write(f"Product extractor not found in {self.model_dir}!")
            prod_tokenizer = None
            prod_extractor = None

        role_model_dir = os.path.join(self.model_dir, "role")
        if os.path.isdir(role_model_dir):
            sys.stderr.write(f"Loading role extractor from {role_model_dir}...")
            config = AutoConfig.from_pretrained(role_model_dir)
            role_tokenizer = AutoTokenizer.from_pretrained(
                role_model_dir,
                use_fast=True
            )
            model_class = (
                BertCRFForRoleLabeling
                if "BertCRFForRoleLabeling" in config.architectures
                else BertForRoleLabeling
            )
            role_extractor = model_class.from_pretrained(
                role_model_dir,
                config=config,
                use_cls=True,
                prod_pooler="span"
            )
            role_labels = ["O"] * len(config.id2label)
            for i, label in config.id2label.items():
                role_labels[int(i)] = label
            sys.stderr.write("done\n")
        else:
            sys.stderr.write("Role labeling model not found in {self.model_dir}!")
            role_tokenizer = None
            role_extractor = None

        self.prod_tokenizer = prod_tokenizer
        self.prod_extractor = prod_extractor.to(self.device)
        self.prod_extractor.eval()
        self.role_tokenizer = role_tokenizer
        self.role_extractor = role_extractor.to(self.device)
        self.role_extractor.eval()
        self.prod_labels = prod_labels
        self.role_labels = role_labels

    def get_products(self, sents):
        """
        """
        # create dataset
        examples = []
        for guid, sent in enumerate(sents):
            # assume sents are not tokenized,
            # todo: replace with better tokenizers
            words = sent.split(" ")
            labels = ["O"] * len(words)
            examples.append(InputExample(
                guid=guid,
                words=words,
                labels=labels
            ))

        features = cre.data.prod.convert_examples_to_features(
            examples,
            self.prod_labels,
            self.prod_max_seq_len,
            self.prod_tokenizer,
            pad_token=self.prod_tokenizer.pad_token_id,
            pad_token_label_id=self.pad_token_label_id
        )

        dataset = RxnDataset(features)
        data_loader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=self.batch_size,
            collate_fn=default_data_collator
        )

        all_preds = []
        for batch in data_loader:
            with torch.no_grad():
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                outputs = self.prod_extractor(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    token_type_ids=batch['token_type_ids']
                )
                logits = outputs[0]

            preds = self.prod_extractor.decode(
                logits,
                batch['decoder_mask'].bool()
            )
            preds = [[self.prod_labels[x] for x in seq] for seq in preds]
            all_preds += preds

        tokenized_sents = [ex.words for ex in examples]
        return tokenized_sents, all_preds

    def get_reactions(self, sents, products=None):
        """
        """
        if products is None:
            tokenized_sents, products = self.get_products(sents)

        assert len(products) == len(tokenized_sents)

        # create dataset
        # for each sent, create #{prod} instances
        examples = []
        num_rxns_per_sent = []
        for guid, (sent, prod_labels) in enumerate(zip(tokenized_sents, products)):
            assert len(sent) == len(prod_labels)
            prods = get_entities(prod_labels)
            num_rxns_per_sent.append(len(prods))
            for i, (etype, ss, se) in enumerate(prods):
                assert etype == "Prod"
                labels = ["O"] * len(sent)
                labels[ss] = "B-Prod"
                labels[ss+1:se+1] = ["I-Prod"] * (se-ss)
                examples.append(InputExample(
                    guid=guid,
                    words=sent,
                    labels=labels
                ))

        features = cre.data.role.convert_examples_to_features(
            examples,
            self.role_labels,
            self.role_max_seq_len,
            self.role_tokenizer,
            pad_token=self.role_tokenizer.pad_token_id,
            pad_token_label_id=self.pad_token_label_id
        )

        dataset = RxnDataset(features)
        data_loader = DataLoader(
            dataset,
            sampler=SequentialSampler(dataset),
            batch_size=self.batch_size,
            collate_fn=default_data_collator
        )

        all_preds = []
        for batch in data_loader:
            with torch.no_grad():
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(self.device)
                outputs = self.role_extractor(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    prod_start_mask=batch['prod_start_mask'],
                    prod_end_mask=batch['prod_end_mask'],
                    prod_mask=batch['prod_mask'],
                    token_type_ids=batch['token_type_ids']
                )
                logits = outputs[0]

            preds = self.role_extractor.decode(
                logits,
                batch['decoder_mask'].bool().to(self.device)
            )
            preds = [[self.role_labels[x] for x in seq] for seq in preds]
            all_preds += preds

        # align predictions with inputs
        example_id = 0
        results = []
        for guid, sent in enumerate(tokenized_sents):
            rxns = {"tokens": sent, "reactions": []}
            for k in range(num_rxns_per_sent[guid]):
                # merge preds with prod labels
                rxn_labels = []
                ex = examples[example_id]
                for j, label in enumerate(ex.labels):
                    if label in ["B-Prod", "I-Prod"]:
                        rxn_labels.append(label)
                    else:
                        rxn_labels.append(all_preds[example_id].pop(0))
                rxn = {}
                for role, ss, se in get_entities(rxn_labels):
                    if role == "Prod":
                        rxn["Product"] = (ss, se)
                    else:
                        if role not in rxn:
                            rxn[role] = [] # e.g., multiple reactants
                        rxn[role].append((ss, se))
                rxns["reactions"].append(rxn)
                example_id += 1

            results.append(rxns)

        return results

