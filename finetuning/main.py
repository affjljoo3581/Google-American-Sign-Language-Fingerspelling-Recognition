# Copyright 2023 Jungwoo Park
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import warnings
from collections import defaultdict
from functools import partial
from typing import Any

import flax
import jax
import Levenshtein
import numpy as np
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.training import checkpoints
from flax.training.common_utils import shard
from torch.utils.data import DataLoader

from dataset import create_train_valid_dataloaders
from training import TrainState, create_train_state, training_step, validation_step

warnings.filterwarnings("ignore")
flax.config.update("flax_use_orbax_checkpointing", False)

save_ckpt = partial(checkpoints.save_checkpoint, "ckpt-finetuning", keep=1)
save_ema_ckpt = partial(checkpoints.save_checkpoint, "ckpt-ema-finetuning", keep=1)


class AverageMeter:
    def __init__(self):
        self.buffer = defaultdict(list)

    def update(self, **kwargs: float):
        for k, v in kwargs.items():
            self.buffer[k].append(v)

    def average(self, prefix: str = "") -> dict[str, float]:
        results = {f"{prefix}{k}": np.mean(v) for k, v in self.buffer.items()}
        self.buffer.clear()
        return results


def evaluate(
    state: TrainState, dataloader: DataLoader, vocab: dict[str, int]
) -> dict[str, Any]:
    single_average_meter, ema_average_meter = AverageMeter(), AverageMeter()
    single_preds, ema_preds, labels = [], [], []
    idx2char = {v: k for k, v in vocab.items()}

    for batch in tqdm.tqdm(dataloader, "evaluate", dynamic_ncols=True, leave=False):
        labels.extend(batch.pop("phrase"))
        batch = shard(jax.tree_map(np.asarray, batch))

        # Run single model to gather metrics and predicted outputs.
        metrics, outputs = validation_step(state, batch, False)
        metrics, outputs = jax.device_get((unreplicate(metrics), outputs))
        single_average_meter.update(**metrics)

        for pred in outputs["preds"].reshape(-1, *outputs["preds"].shape[2:]):
            single_preds.append("".join(idx2char.get(i, "") for i in pred))

        # Run ema-decayed model to gather metrics and predicted outputs.
        metrics, outputs = validation_step(state, batch, True)
        metrics, outputs = jax.device_get((unreplicate(metrics), outputs))
        ema_average_meter.update(**metrics)

        for pred in outputs["preds"].reshape(-1, *outputs["preds"].shape[2:]):
            ema_preds.append("".join(idx2char.get(i, "") for i in pred))

    # Aggregate the label lengths and edit distances to calculate normalized edit
    # distance score. Suppose `N` is the sum of label lengths and `D` is the sum of edit
    # distances, then the score is `(N - D) / N`.
    N = sum(map(len, labels))
    D_single = sum(Levenshtein.distance(p, l) for p, l in zip(single_preds, labels))
    D_ema = sum(Levenshtein.distance(p, l) for p, l in zip(ema_preds, labels))

    return {
        **single_average_meter.average("val/"),
        **ema_average_meter.average("val/ema_"),
        "val/preds": wandb.Table(["pred", "label"], list(zip(single_preds, labels))),
        "val/score": (N - D_single) / N,
        "val/ema_preds": wandb.Table(["pred", "label"], list(zip(ema_preds, labels))),
        "val/ema_score": (N - D_ema) / N,
    }


def main(args: argparse.Namespace):
    jax.distributed.initialize()
    train_dataloader, valid_dataloader, vocab = create_train_valid_dataloaders(args)

    state = create_train_state(args, vocab).replicate()
    wandb.init(name=args.name, project=args.project, config=args)

    average_meter, current_epoch = AverageMeter(), 0
    best_val_score, best_val_ema_score = -np.inf, -np.inf
    progress_bar = tqdm.trange(args.training_steps, dynamic_ncols=True)

    while progress_bar.n < args.training_steps:
        for batch in train_dataloader:
            if (step := progress_bar.n) >= args.training_steps:
                break

            batch.pop("phrase")
            batch = shard(jax.tree_map(np.asarray, batch))

            state, metrics = training_step(state, batch)
            average_meter.update(**jax.device_get(unreplicate(metrics)))
            progress_bar.update()

            # Log the training metrics, learning rate, and current epoch.
            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                metrics = {
                    **average_meter.average("train/"),
                    "train/learning_rate": state.learning_rate_fn(step),
                    "epoch": current_epoch,
                }
                wandb.log(metrics, step)

            # Evaluate the model performance on validation set and save the best-scored
            # model checkpoint.
            if args.val_interval > 0 and (step + 1) % args.val_interval == 0:
                wandb.log((metrics := evaluate(state, valid_dataloader, vocab)))

                if metrics["val/score"] > best_val_score:
                    save_ckpt(unreplicate(state.params), step + 1)
                    best_val_score = metrics["val/score"]

                if metrics["val/ema_score"] > best_val_ema_score:
                    save_ema_ckpt(unreplicate(state.ema_params), step + 1)
                    best_val_ema_score = metrics["val/ema_score"]
        current_epoch += 1

    # Log the highest validation score to compare performances directly in WandB page.
    wandb.log({"val/best_score": max(best_val_score, best_val_ema_score)})


if __name__ == "__main__":
    DEFAULT_TRAIN_FILEPATH = "resources/competition/train_landmarks_npy/{}.npy"
    DEFAULT_TRAIN_LABEL_PATH = "resources/competition/train.csv"
    DEFAULT_SUPP_FILEPATH = "resources/competition/supplemental_landmarks_npy/{}.npy"
    DEFAULT_SUPP_LABEL_PATH = "resources/competition/supplemental_metadata.csv"
    DEFAULT_ALIGNMENTS_PATH = "resources/competition/alignments.csv"
    DEFAULT_VOCAB_PATH = "resources/competition/character_to_prediction_index.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-filepath", default=DEFAULT_TRAIN_FILEPATH)
    parser.add_argument("--train-labels", default=DEFAULT_TRAIN_LABEL_PATH)
    parser.add_argument("--supp-filepath", default=DEFAULT_SUPP_FILEPATH)
    parser.add_argument("--supp-labels", default=DEFAULT_SUPP_LABEL_PATH)
    parser.add_argument("--use-supplemental", action="store_true", default=False)

    parser.add_argument("--alignments", default=DEFAULT_ALIGNMENTS_PATH)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--use-cutmix", action="store_true", default=False)

    parser.add_argument("--num-workers", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--label-length", type=int, default=64)
    parser.add_argument("--input-features", type=int, default=225)

    parser.add_argument("--init-seed", type=int, default=2023)
    parser.add_argument("--dropout-seed", type=int, default=2024)
    parser.add_argument("--hard-distill", action="store_true", default=False)
    parser.add_argument("--pretrained-ckpt")
    parser.add_argument("--teacher-ckpt")

    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--dim", type=int, default=256)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--layerdrop", type=float, default=0.1)
    parser.add_argument("--use-lstm-head", action="store_true", default=False)

    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--adam-eps", type=float, default=1e-6)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--ema-decay", type=float, default=0.993)

    parser.add_argument("--name", default="transformer-6l-256d")
    parser.add_argument("--project", default="gaslfr-data2vec2-finetuning")
    parser.add_argument("--warmup-steps", type=int, default=1000)
    parser.add_argument("--training-steps", type=int, default=70000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--val-interval", type=int, default=1000)
    main(parser.parse_args())
