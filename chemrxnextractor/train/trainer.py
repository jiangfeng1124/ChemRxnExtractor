import logging
import math
import os
import re
from typing import Any, Callable, Optional
from typing import Dict, List, Tuple, Union

import numpy as np
import torch
from torch import nn
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm.auto import tqdm, trange

from transformers import Trainer
from transformers.modeling_utils import PreTrainedModel
from transformers.training_args import TrainingArguments
from transformers.data.data_collator import DataCollator
from transformers.optimization import AdamW, get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


class IETrainer(Trainer):
    """
    IETrainer is inheritated from from transformers.Trainer, optimized for IE tasks.
    """
    def __init__(
        self,
        model: PreTrainedModel,
        args: TrainingArguments,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        compute_metrics=None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = None,
        use_crf: Optional[bool]=False
    ):
        super(IETrainer, self).__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            optimizers=optimizers
        )
        self.use_crf = use_crf

    def get_optimizers(
        self,
        num_training_steps: int
    ) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]:
        """
        Setup the optimizer and the learning rate scheduler.
        """
        if self.optimizers is not None:
            return self.optimizers

        no_decay = ["bias", "LayerNorm.weight"]
        if self.use_crf:
            crf = "crf"
            crf_lr = self.args.crf_learning_rate
            logger.info(f"Learning rate for CRF: {crf_lr}")
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if (not any(nd in n for nd in no_decay)) and (crf not in n)
                    ],
                    "weight_decay": self.args.weight_decay
                },
                {
                    "params": [p for p in self.model.crf.parameters()],
                    "weight_decay": self.args.weight_decay,
                    "lr": crf_lr
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay) and (not crf not in n)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [
                        p for n, p in self.model.named_parameters()
                        if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]

        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.args.learning_rate,
            eps=self.args.adam_epsilon
        )
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=num_training_steps
        )

        return optimizer, scheduler

    def evaluate(self, eval_dataset: Optional[Dataset] = None) -> Dict:
        eval_dataloader = self.get_eval_dataloader(eval_dataset)

        if self.use_crf:
            output = self._prediction_loop_crf(eval_dataloader, description="Evaluation")
        else:
            output = self._prediction_loop(eval_dataloader, description="Evaluation")

        self._log(output['metrics'])

        return output

    def predict(self, test_dataset: Dataset) -> Dict:
        test_dataloader = self.get_test_dataloader(test_dataset)

        if self.use_crf:
            return self._prediction_loop_crf(test_dataloader, description="Prediction")
        else:
            return self._prediction_loop(test_dataloader, description="Prediction")

    def _prediction_loop_crf(
        self,
        dataloader: DataLoader,
        description: str
    ) -> Dict:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`
        Works both with or without labels.
        """
        model = self.model
        batch_size = dataloader.batch_size

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        eval_losses: List[float] = []
        preds_ids = []
        label_ids = []

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs.get(k) is not None
                for k in ["labels", "lm_labels", "masked_lm_labels"]
            )

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            mask = inputs["decoder_mask"].to(torch.bool)
            preds = model.decode(logits, mask=mask)
            preds_ids.extend(preds)
            if inputs.get("labels") is not None:
                labels = [inputs["labels"][i, mask[i]].tolist() \
                            for i in range(inputs["labels"].shape[0])]
                label_ids.extend(labels)
                assert len(preds) == len(labels)
                assert len(preds[0]) == len(labels[0])

        if self.compute_metrics is not None and \
                len(preds_ids) > 0 and \
                len(label_ids) > 0:
            metrics = self.compute_metrics(preds_ids, label_ids)
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics['eval_loss'] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return {'predictions': preds_ids, 'label_ids': label_ids, 'metrics': metrics}

    def _prediction_loop(
        self,
        dataloader: DataLoader,
        description: str
    ) -> Dict:
        """
        Prediction/evaluation loop, shared by `evaluate()` and `predict()`.
        Works both with or without labels.
        """
        model = self.model
        batch_size = dataloader.batch_size

        logger.info("***** Running %s *****", description)
        logger.info("  Num examples = %d", self.num_examples(dataloader))
        logger.info("  Batch size = %d", batch_size)

        model.eval()

        eval_losses: List[float] = []
        preds: List[torch.Tensor] = []
        label_ids: List[torch.Tensor] = []

        for inputs in tqdm(dataloader, desc=description):
            has_labels = any(
                inputs.get(k) is not None
                for k in ["labels", "lm_labels", "masked_lm_labels"]
            )

            for k, v in inputs.items():
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.args.device)

            with torch.no_grad():
                outputs = model(**inputs)
                if has_labels:
                    step_eval_loss, logits = outputs[:2]
                    eval_losses += [step_eval_loss.mean().item()]
                else:
                    logits = outputs[0]

            preds.append(logits.detach().cpu())

            if inputs.get("labels") is not None:
                label_ids.append(inputs["labels"].detach().cpu())

        preds = torch.cat(preds, dim=0)
        label_ids = torch.cat(label_ids, dim=0) if len(label_ids) > 0 else None

        if self.compute_metrics is not None and \
                preds is not None and \
                label_ids is not None:
            metrics = self.compute_metrics(preds, label_ids)
        else:
            metrics = {}
        if len(eval_losses) > 0:
            metrics["eval_loss"] = np.mean(eval_losses)

        # Prefix all keys with eval_
        for key in list(metrics.keys()):
            if not key.startswith("eval_"):
                metrics[f"eval_{key}"] = metrics.pop(key)

        return {'predictions': preds, 'label_ids': label_ids, 'metrics': metrics}

