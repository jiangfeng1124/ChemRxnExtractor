import logging
from chemrxnextractor.models import BertForTagging
from chemrxnextractor.models import BertCRFForTagging
from chemrxnextractor.models import BertForRoleLabeling
from chemrxnextractor.models import BertCRFForRoleLabeling
from chemrxnextractor.data.utils import InputExample
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler
from transformers.data.data_collator import default_data_collator


logger = logging.getLogger(__name__)

class RxnDataset(Dataset):
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return len(self.features)

    def __getitem__(self, i):
        return self.features[i]


class Extractor(object):
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
            config = AutoConfig.from_pretrained(prod_model_dir)
            prod_tokenizer = AutoTokenizer.from_pretrained(
                prod_model_dir,
                use_fast=True
            )
            model_class = (
                BertCRFForRoleLabeling
                if "BertCRFForTagging" in config.architectures
                else BertForTagging
            )
            prod_extractor = model_class.from_pretrained(
                prod_model_dir,
                config=config
            )
            prod_labels = [] * len(config.id2label)
            for i, label in id2label.items():
                prod_labels[int(i)] = label
        else:
            logger.info(f"Product extractor not found in {self.model_dir}!")
            prod_tokenizer = None
            prod_extractor = None

        role_model_dir = os.path.join(self.model_dir, "role")
        if os.path.isdir(role_model_dir):
            config = AutoConfig.from_pretrained(role_model_dir)
            role_tokenizer = AutoTokenizer.from_pretrained(
                rold_model_dir,
                use_fast=True
            )
            model_class = (
                BertCRFForRoleLabeling
                if "BertCRFForTagging" in config.architectures
                else BertForTagging
            )
            role_extractor = model_class.from_pretrained(
                role_model_dir,
                config=config
            )
            role_labels = [] * len(config.id2label)
            for i, label in id2label.items():
                role_labels[int(i)] = label
        else:
            logger.info("Role labeling model not found in {self.model_dir}!")
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
            words = sent.split(" ")
            examples.append(InputExample(guid=guid, words=words))

        features = convert_examples_to_features(
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
                outputs = self.prod_extractor(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device)
                )
                logits = outputs[0]

            preds = self.prod_extractor.decode(
                logits,
                batch['decoder_mask'].bool().to(self.device)
            )
            preds = [[self.prod_labels[x] for x in seq] for seq in preds]
            all_preds += preds

        return all_preds

    def get_reactions(self, sents, products=None):
        """
        """
        if products is None:
            logging.info("Extracting products...")
            products = self.get_products(sents)

        assert len(products) == len(sents)

        # create dataset
        # for each sent, create #{prod} instances
        examples = []
        for guid, (sent, prod_labels) in enumerate(zip(sents, products)):
            words = sent.split(" ")
            assert len(words) == len(prod_labels)
            prods = get_entities(prod_labels)
            for i, (etype, ss, se) in enumerate(prods):
                assert etype == "Prod"
                labels = ["O"] * len(words)
                labels[ss] = "B-Prod"
                labels[ss+1:se+1] = ["I-Prod"] * (se-ss)
                examples.append(InputExample(
                    guid=guid,
                    words=words,
                    labels=labels
                ))

        features = convert_examples_to_features(
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
                outputs = self.role_extractor(
                    input_ids=batch['input_ids'].to(self.device),
                    attention_mask=batch['attention_mask'].to(self.device),
                    token_type_ids=batch['token_type_ids'].to(self.device)
                )
                logits = outputs[0]

            preds = self.role_extractor.decode(
                logits,
                batch['decoder_mask'].bool().to(self.device)
            )
            preds = [[self.role_labels[x] for x in seq] for seq in preds]
            all_preds += preds

        results = {}
        assert len(examples) == len(all_preds)
        for ex, preds in zip(examples, all_preds):
            guid = ex.guid # sent id
            # merge preds with ex.labels
            rxn_labels = []
            for j, label in enumerate(ex.labels):
                if label in ["B-Prod", "I-Prod"]:
                    rxn_labels.append(label)
                else:
                    if preds:
                        rxn_labels.append(preds.pop(0))
                    else:
                        logger.info(f"No prediction for {ex.words[j]}")
            results[guid].append(rxn_labels)

        return results


if __name__ == '__main__':
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", type=str)
    parser.add_argument("--input", type=str)
    args = parser.parse_args()

    rxn_extractor = Extractor(model_dir=args.model_dir)

    with open(args.input, "r") as f:
        sents = f.read().splitlines()
    rxns = rxn_extractor.get_reactions(sents)
    print(rxns)

