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
import os
import random
import time
import torch
import clip
import numpy as np
import pickle as pkl

from pc.clip_classifier import get_clip_classifier
from datetime import datetime
from PIL import Image
from typing import List, Tuple, Dict, Set, Any, Optional, Callable
from pytorch_transformers import BertTokenizer, BertForSequenceClassification
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
from tensorboardX import SummaryWriter
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


class ClipDataset(Dataset):
    def __init__(self, task: Task, image_preprocesser, train: bool, gan_imgs: bool=False, text_only: bool=False, dev: bool=True) -> None:
        """
        Args:
            task: task to use
            train: True for train, False for test
        """
        # load labels and y data
        self.text_only = text_only
        train_data, test_data = get(task, dev)
        split_data = train_data if train else test_data
        self.labels, self.y = split_data
        assert len(self.labels) == len(self.y)

        # load X index
        # line_mapping maps from word1/word2 label to sentence index in sentence list.
        self.line_mapping = {}
        task_short = TASK_SHORTHAND[task]
        with open("data/sentences/index.csv", "r") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                if row["task"] == task_short:
                    self.line_mapping[row["uids"]] = i
                    # TODO: check that i lines up and isn't off by one

        with open("data/sentences/sentences.txt", "r") as f:
            all_sentences = [line.strip() for line in f.readlines()]
            self.sentences = [all_sentences[self.line_mapping[label]] for label in self.labels]
            self.tokenized_sents = clip.tokenize(self.sentences)

        # Load map from sentence index to image names and get list of image names
        if not text_only:
            if gan_imgs:
                self.images = [f"data/situated_sentence_images/{self.line_mapping[label]}.png" for label in self.labels]
            else:
                sent_idx_to_image = pkl.load(open("data/clip/sent_idx_to_image.pkl", "rb"))[task]
                self.images = ["data/mscoco/images/{}".format(sent_idx_to_image[self.line_mapping[label]]) for label in self.labels]

        # show some samples. This is a really great idiom that huggingface does. Baking
        # little visible sanity checks like this into your code is just... *does gesture
        # where you kiss your fingers and throw them away from your mouth as if
        # describing great food.*
        n_sample = 5
        print("{} Samples:".format(n_sample))
        for i in random.sample(range(len(self.labels)), n_sample):
            label = self.labels[i]
            sentence = self.sentences[i]
            if not text_only:
                image = self.images[i]
                print('- {}: "{}", "{}"'.format(label, sentence, image))
            else:
                print('- {}: "{}"'.format(label, sentence))

        if not text_only:
            self.image_preprocesser = image_preprocesser

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        label = self.labels[i]
        tok_sent = self.tokenized_sents[i]

        if not self.text_only:
            image_name = self.images[i]

            # process image
            image = self.image_preprocesser(Image.open(image_name))

            return {
                "input_ids": tok_sent,
                "input_image": image,
                "label": label,
                "y": self.y[i],
            }

        else:
            return {
                "input_ids": tok_sent,
                "label": label,
                "y": self.y[i],
            }



def make_epoch_runner(
    task: Task, device: Any, model: nn.Module, loss_fn: Any, optimizer: Any, scheduler: Any, viz: Any, text_only: bool = False
):
    """This closure exists so we can duplicate code less."""

    def epoch(
        loader: DataLoader, data_len: int, train: bool, split: str, global_i: int, text_only: bool = False
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
            y = batch["y"].to(device, dtype=torch.half)
            input_ids = batch["input_ids"].to(device)

            if not text_only:
                input_images = batch["input_image"].to(device)

            labels += batch["label"]
            batch_size = len(y)

            # fwd
            if train:
                if not text_only:
                    y_hat = model(text=input_ids, image=input_images)
                else:
                    y_hat = model(text=input_ids)

                loss = loss_fn(y_hat, y)
                loss.backward()
                scheduler.step()
                optimizer.step()
                model.zero_grad()
                global_i += batch_size
            else:
                with torch.no_grad():
                    if not text_only:
                        y_hat = model(text=input_ids, image=input_images)
                    else:
                        y_hat = model(text=input_ids)

                    loss = loss_fn(y_hat, y)

            batch_decisions = torch.tensor([int(value >= .5) for value in y_hat]).to(device)
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

        return metrics_results, model

    return epoch


def main() -> None:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--gan-imgs", dest="gan_imgs", action="store_true")
    parser.add_argument("--text-only", dest="text_only", action="store_true")
    parser.add_argument("--dev", dest="dev", action="store_true")
    parser.add_argument(
        "--task",
        type=str,
        choices=TASK_REV_MEDIUMHAND.keys(),
        help="Name of task to run",
        required=True,
    )
    parser.add_argument("--epochs", type=int, default=5, help="How many epochs to run")
    parser.add_argument("--early-stopping", type=int, default=5, help="num runs with less than threshold change in loss")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--optimizer", type=str, default="adamw", help="training optimizer")
    parser.add_argument("--warmup-ratio", type=float, default="0.1", help="training optimizer")
    parser.add_argument("--batch-size", type=int, default=64, help="training batch size")
    parser.add_argument("--activation", type=str, default="relu", help="mlp hidden layer activation")
    parser.add_argument("--dropout", type=float, default=0.0, help="mlp drop out")
    args = parser.parse_args()
    task = TASK_REV_MEDIUMHAND[args.task]

    device = torch.device("cuda")
    initial_lr = args.lr
    warmup_proportion = args.warmup_ratio
    train_batch_size = args.batch_size
    test_batch_size = 96
    train_epochs = args.epochs

    if args.activation.lower() == "relu":
        activation = nn.ReLU
    elif args.activation.lower() == "gelu":
        activation = nn.GELU
    else:
        raise Exception("Please give a valid activation function. One of relu or gelu")

    if args.dev:
        dev=True
    else:
        dev=False

    print("Building model...")
    if args.text_only:
        model, preprocess = get_clip_classifier(512, args.dropout, 128, activation, args.dropout, 1, text_only=args.text_only)
        model.to(device)
    else:
        model, preprocess = get_clip_classifier(1024, args.dropout, 128, activation, args.dropout, 1, text_only=args.text_only)
        model.to(device)

    print("Loading traning data")
    train_dataset = ClipDataset(task, preprocess, True, gan_imgs=args.gan_imgs, text_only=args.text_only, dev=dev)
    train_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True, num_workers=8
    )
    print("Loading test data")
    test_dataset = ClipDataset(task, preprocess, False, gan_imgs=args.gan_imgs, text_only=args.text_only, dev=dev)
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
            "weight_decay": 0.0,
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
    loss_fn = nn.MSELoss()

    if args.optimizer.lower() == "adamw":
        optimizer = AdamW(optimizer_grouped_parameters, lr=initial_lr)
    elif args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=initial_lr)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(optimizer_grouped_parameters, lr=initial_lr)
    else:
        raise Exception("Please give a valid optimizer. One of adamw, adam, or sgd")
        
    scheduler = WarmupLinearSchedule(
        optimizer, warmup_steps=warmup_proportion * t_total, t_total=t_total
    )

    run_time_str = datetime.now().strftime("%y-%m-%d_%H-%M-%S")
    viz = SummaryWriter("runs/clip/{}/{}".format(args.task, run_time_str))

    if args.gan_imgs:
        image_state = "gan_imgs"
    elif args.text_only:
        image_state = "text_only"
    else:
        image_state = "mscoco_imgs"

    global_i = 0
    epoch = make_epoch_runner(task, device, model, loss_fn, optimizer, scheduler, viz, args.text_only)
    # print("Running eval before training.")
    # epoch(test_loader, len(test_dataset), False, "test", global_i)
    for epoch_i in range(train_epochs):
        print("Starting epoch {}/{}.".format(epoch_i + 1, train_epochs))
        epoch(train_loader, len(train_dataset), True, "train", global_i, args.text_only)
        global_i += len(train_dataset)
    print("Running eval after {} epochs.".format(train_epochs))
    metrics_results, model = epoch(test_loader, len(test_dataset), False, "test", global_i, args.text_only)

    if args.dev:
        # Save run scores with hyperparams specified 
        dir_name = f"runs/clip/{args.task}-hyperparam-search"
        _, micro_f1, macro_f1s, _, _ = metrics_results
        if not os.path.exists(dir_name):
            os.mkdir(dir_name)

        with open(f"{dir_name}/{args.lr}-warmup{args.warmup_ratio}-{args.activation}-dropout{args.dropout}", "w") as r:
            r.write(f"micro f1: {micro_f1} ")
            for cat, macro_f1 in macro_f1s.items():
                r.write(f"{cat}: {macro_f1} ")

            r.close()

    else:
        #Save model and results
        if not os.path.exists(f"runs/clip/{args.task}"):
            os.mkdir(f"runs/clip/{args.task}")
        torch.save(model, f"runs/clip/{args.task}/{image_state}_clip_classifier.pt")
        #model.save(f"runs/clip/{args.task}/{image_state}_clip_classifier.pt")

        # write per-datum results to file
        _, micro_f1, macro_f1s, _, per_datum = metrics_results
        
        with open(f"runs/clip/{args.task}/{image_state}_test_results.txt", "w") as r:
            r.write(f"micro f1: {micro_f1} ")
            for cat, macro_f1 in macro_f1s.items():
                r.write(f"{cat}: {macro_f1} ")

            r.close()

        path = os.path.join(
            "data", "results", "{}-{}-perdatum.txt".format("clip", TASK_MEDIUMHAND[task])
        )
        with open(path, "w") as f:
            f.write(util.np2str(per_datum) + "\n")

        viz.flush()

if __name__ == "__main__":
    main()
