"""Bert model.

This code is all separate from the rest of the experiment code because Bert is run
differently than all other models. It used to be run the same, but it did worse than
ELMo, which is not the Bert Way. Now, finetuned end-to-end as is the Bert Way, it wins
across the board. (Which is also the Bert Way.)

Note that Bert, at least in this task, is exteremely volatile. Some runs will just fail
and give zero F1 score.

Epochs needed:
- abstract OP: 5
- situated OP: 5
- situated OA: 5
- situated AP: 1

Much of the finetuning gunk code itself (in main()) gratefully adapted from:
https://github.com/huggingface/pytorch-transformers/
"""

import argparse
import code  # code.interact(local=dict(globals(), **locals()))
import csv
from datetime import datetime
import os
import random
import time
from typing import List, Tuple, Dict, Set, Any, Optional, Callable

import numpy as np
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from tensorboardX import SummaryWriter
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from pc.data import (
    Task,
    get,
    TASK_SHORTHAND,
    TASK_MEDIUMHAND,
    TASK_REV_MEDIUMHAND,
    TASK_LABELS,
)
from pc import metrics
from pc import util

torch.manual_seed(2021)

class BertDataset(Dataset):
    def __init__(self, task: Task, train: bool, seq_len: int = 20, dev: bool = True) -> None:
        """
        Args:
            task: task to use
            train: True for train, False for test
            seq_len: sequence length. Set to 2 higher than you need for tokens to
                account for [CLS] and [SEP]
        """
        self.seq_len = seq_len

        # load labels and y data
        train_data, test_data = get(task, dev)
        split_data = train_data if train else test_data
        labels, y_list = split_data
        self.label_to_y = {}
        for label, y in zip(labels, y_list):
            self.label_to_y[label] = y

        #self.labels, self.y = split_data
        #assert len(self.labels) == len(self.y)

        # load X index
        # line_mapping maps from word1/word2 label to sentence index in sentence list.
        self.line_mapping: Dict[str, int] = {}
        self.line_mapping_r = {}
        task_short = TASK_SHORTHAND[task]
        with open("data/sentences/index.csv", "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if row["task"] == task_short:
                    self.line_mapping[row["uids"]] = i
                    self.line_mapping_r[i] = row["uids"]
                    # TODO: check that i lines up and isn't off by one

        self.task_idxs = sorted(list(set([self.line_mapping[label] for label in labels])))

        with open("data/sentences/sentences.txt", "r") as f:
            all_sentences = [line.strip() for line in f.readlines()]
            self.sent_idx_to_dataset_id = {}
            self.sentences = []
            for i, sent in enumerate(all_sentences):
                if i in self.task_idxs:
                    self.sentences.append(sent)
                    self.sent_idx_to_dataset_id[i] = len(self.sentences) - 1

            #self.sentences = [line.strip() for line in f.readlines()]


        # show some samples. This is a really great idiom that huggingface does. Baking
        # little visible sanity checks like this into your code is just... *does gesture
        # where you kiss your fingers and throw them away from your mouth as if
        # describing great food.*
        n_sample = 5
        print("{} Samples:".format(n_sample))
        for i in random.sample(range(len(self.task_idxs)), n_sample):
            #label = self.labels[i]
            label = self.line_mapping_r[self.task_idxs[i]]
            sentence = self.sentences[i]
            #sentence = self.sentences[self.line_mapping[label]]
            print('- {}: "{}"'.format(label, sentence))

        print("Loading tokenizer...")
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-large-uncased", do_lower_case=True, do_basic_tokenize=True
        )

    def __len__(self) -> int:
        #return len(self.labels)
        return len(self.task_idxs)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        label = self.line_mapping_r[self.task_idxs[i]]
        #label = self.labels[i]

        # tokenize
        max_sent_len = self.seq_len - 2
        sentence = self.sentences[i]
        tkns = ["[CLS]"] + self.tokenizer.tokenize(sentence)[:max_sent_len] + ["[SEP]"]

        input_mask = [1] * len(tkns)

        # pad
        if len(tkns) < self.seq_len:
            pad_len = self.seq_len - len(tkns)
            tkns += ["[PAD]"] * pad_len
            input_mask += [0] * pad_len

        # code.interact(local=dict(globals(), **locals()))
        input_ids = self.tokenizer.convert_tokens_to_ids(tkns)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "input_mask": torch.tensor(input_mask, dtype=torch.long),
            "label": label,
            "y": self.label_to_y[label],
        }


def make_epoch_runner(
    task: Task, device: Any, model: nn.Module, optimizer: Any, scheduler: Any, viz: Any
):
    """This closure exists so we can duplicate code less."""

    def epoch(
        loader: DataLoader, data_len: int, train: bool, split: str, global_i: int
    ) -> Tuple[float, float, Dict[str, float], Dict[int, Dict[str, Any]], np.ndarray]:
        """
        Returns results of metrics.report(...)
        """
        model.train(train)
        labels: List[str] = []
        total_corr, total_loss, start_idx = 0, 0, 0
        epoch_y_hat = np.zeros(data_len, dtype=int)
        epoch_y = np.zeros(data_len, dtype=int)
        for batch_i, batch in enumerate(tqdm(loader, desc="Batch")):
            y = batch["y"].to(device, dtype=torch.long)
            input_ids = batch["input_ids"].to(device)
            input_mask = batch["input_mask"].to(device)
            labels += batch["label"]
            batch_size = len(y)

            # fwd
            if train:
                outputs = model(
                    input_ids=input_ids, attention_mask=input_mask, labels=y
                )
                loss, logits = outputs[:2]
                loss.backward()
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_i += batch_size
            else:
                with torch.no_grad():
                    outputs = model(
                        input_ids=input_ids, attention_mask=input_mask, labels=y
                    )
                loss, logits = outputs[:2]

            batch_decisions = logits.argmax(dim=1)
            batch_corr = (batch_decisions == y).sum().item()
            total_corr += batch_corr
            total_loss += loss.item() * batch_size
            batch_acc = batch_corr / batch_size

            epoch_y_hat[start_idx : start_idx + batch_size] = (
                batch_decisions.int().cpu().numpy()
            )
            epoch_y[start_idx : start_idx + batch_size] = (
                y.int().cpu().squeeze().numpy()
            )

            # viz per-batch stats for training only
            if train:
                viz.add_scalar("Loss/{}".format(split), loss.item(), global_i)
                viz.add_scalar("Acc/{}".format(split), batch_acc, global_i)

            start_idx += batch_size

        # end of batch. always print overall stats.
        avg_loss = total_loss / data_len
        overall_acc = total_corr / data_len
        print("Average {} loss: {}".format(split, avg_loss))
        print("{} accuracy: {}".format(split, overall_acc))

        # for eval only, viz overall loss and acc
        if not train:
            viz.add_scalar("Loss/{}".format(split), avg_loss, global_i)
            viz.add_scalar("Acc/{}".format(split), overall_acc, global_i)

        # for both train and eval, compute overall stats.
        assert len(labels) == len(epoch_y_hat)
        # code.interact(local=dict(globals(), **locals()))
        metrics_results = metrics.report(
            epoch_y_hat, epoch_y, labels, TASK_LABELS[task]
        )
        _, micro_f1, category_macro_f1s, _, _ = metrics_results
        viz.add_scalar("F1/{}/micro".format(split), micro_f1, global_i)
        for cat, macro_f1 in category_macro_f1s.items():
            viz.add_scalar("F1/{}/macro/{}".format(split, cat), macro_f1, global_i)
        viz.flush()

        return metrics_results

    return epoch


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--dev", dest="dev", action="store_true")
    parser.add_argument(
        "--task",
        type=str,
        choices=TASK_REV_MEDIUMHAND.keys(),
        help="Name of task to run",
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=5, help="How many epochs to run")
    args = parser.parse_args()
    task = TASK_REV_MEDIUMHAND[args.task]

    device = torch.device("cuda")
    initial_lr = 5e-5
    warmup_proportion = 0.1
    train_batch_size = 64
    test_batch_size = 96
    train_epochs = args.epochs

    print("Building model...")
    model = BertForSequenceClassification.from_pretrained(
        "bert-large-uncased", num_labels=2
    )
    model.to(device)

    print("Loading traning data")
    train_dataset = BertDataset(task, True, dev=args.dev)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8
    )
    print("Loading test data")
    test_dataset = BertDataset(task, False, dev=args.dev)
    test_loader = DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False, num_workers=8
    )

    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    t_total = int(((len(train_dataset) // train_batch_size) + 1) * train_epochs)
    print("Num train optimization steps: {}".format(t_total))
    optimizer = AdamW(optimizer_grouped_parameters, lr=initial_lr)
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_proportion * t_total, t_total=t_total
    )

    run_time_str = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    viz = SummaryWriter("runs/bert/{}/{}".format(args.task, run_time_str))

    global_i = 0
    epoch = make_epoch_runner(task, device, model, optimizer, scheduler, viz)
    # print("Running eval before training.")
    # epoch(test_loader, len(test_dataset), False, "test", global_i)
    for epoch_i in range(train_epochs):
        print("Starting epoch {}/{}.".format(epoch_i + 1, train_epochs))
        epoch(train_loader, len(train_dataset), True, "train", global_i)
        global_i += len(train_dataset)
    print("Running eval after {} epochs.".format(train_epochs))
    metrics_results = epoch(test_loader, len(test_dataset), False, "test", global_i)
#    viz.flush()
#
#    # write per-datum results to file
#    _, _, _, _, per_datum = metrics_results
#    path = os.path.join(
#        "data", "results", "{}-{}-perdatum.txt".format("Bert", TASK_MEDIUMHAND[task])
#    )
#    with open(path, "w") as f:
#        f.write(util.np2str(per_datum) + "\n")

    if args.dev:
        # Save run scores with hyperparams specified
        dir_name = f"runs/bert/{args.task}-hyperparam-search"
        _, micro_f1, macro_f1s, _, _ = metrics_results
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        with open(f"{dir_name}/dev_results.txt", "w") as r:
            r.write(f"micro f1: {micro_f1} ")
            for cat, macro_f1 in macro_f1s.items():
                r.write(f"{cat}: {macro_f1} ")

            r.close()

    else:
        #Save model and results
        if not os.path.exists(f"runs/bert/{args.task}"):
            os.mkdir(f"runs/bert/{args.task}")
        torch.save(model, f"runs/bert/{args.task}/bert_classifier.pt")
        #model.save(f"runs/clip/{args.task}/{image_state}_clip_classifier.pt")

        # write per-datum results to file
        _, micro_f1, macro_f1s, _, per_datum = metrics_results

        with open(f"runs/bert/{args.task}/test_results.txt", "w") as r:
            r.write(f"micro f1: {micro_f1} ")
            for cat, macro_f1 in macro_f1s.items():
                r.write(f"{cat}: {macro_f1} ")

            r.close()

        path = os.path.join(
            "data", "results", "{}-{}-perdatum.txt".format(f"bert".replace("_", "-"), TASK_MEDIUMHAND[task])
        )
        with open(path, "w") as f:
            f.write(util.np2str(per_datum) + "\n")

        viz.flush()


if __name__ == "__main__":
    main()
