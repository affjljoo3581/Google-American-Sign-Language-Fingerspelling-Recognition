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

import flax
import jax
import numpy as np
import tqdm
import wandb
from flax.jax_utils import unreplicate
from flax.training import checkpoints
from flax.training.common_utils import shard

from dataset import create_train_dataloader
from training import create_train_state, training_step

warnings.filterwarnings("ignore")
flax.config.update("flax_use_orbax_checkpointing", False)


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


def main(args: argparse.Namespace):
    jax.distributed.initialize()
    train_dataloader = create_train_dataloader(args)

    state = create_train_state(args).replicate()
    wandb.init(name=args.name, project=args.project, config=args)

    average_meter, current_epoch = AverageMeter(), 0
    progress_bar = tqdm.trange(args.training_steps, dynamic_ncols=True)

    while progress_bar.n < args.training_steps:
        for batch in train_dataloader:
            if (step := progress_bar.n) >= args.training_steps:
                break

            state, metrics = training_step(state, shard(batch.numpy()))
            average_meter.update(**jax.device_get(unreplicate(metrics)))
            progress_bar.update()

            # Log the training metrics, learning rate, and current epoch.
            if args.log_interval > 0 and (step + 1) % args.log_interval == 0:
                metrics = {
                    **average_meter.average("train/"),
                    "train/learning_rate": state.learning_rate_fn(step),
                    "train/ema_decay": state.ema_decay_fn(step),
                    "epoch": current_epoch,
                }
                wandb.log(metrics, step)

            if args.ckpt_interval > 0 and (step + 1) % args.ckpt_interval == 0:
                checkpoints.save_checkpoint(
                    ckpt_dir="ckpt-pretraining",
                    target=unreplicate(state.params),
                    step=step + 1,
                    keep=args.num_keep_ckpt,
                )
        current_epoch += 1

    # Save the latest weights to the checkpoint file.
    if args.ckpt_interval > 0 and step % args.ckpt_interval != 0:
        checkpoints.save_checkpoint(
            ckpt_dir="ckpt-pretraining",
            target=unreplicate(state.params),
            step=args.training_steps,
            keep=args.num_keep_ckpt,
        )


if __name__ == "__main__":
    DEFAULT_LABEL_PATH = "resources/competition/train.csv"

    parser = argparse.ArgumentParser()
    parser.add_argument("--filenames", default="resources/competition/*/*.npy")
    parser.add_argument("--labels", default=DEFAULT_LABEL_PATH)
    parser.add_argument("--num-workers", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--min-length", type=int, default=10)

    parser.add_argument("--encoder-layers", type=int, default=6)
    parser.add_argument("--encoder-dim", type=int, default=256)
    parser.add_argument("--encoder-heads", type=int, default=4)
    parser.add_argument("--encoder-dropout", type=float, default=0.1)
    parser.add_argument("--encoder-layerdrop", type=float, default=0.1)

    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--decoder-dim", type=int, default=256)
    parser.add_argument("--decoder-kernel", type=int, default=7)
    parser.add_argument("--decoder-dropout", type=float, default=0.0)
    parser.add_argument("--decoder-layerdrop", type=float, default=0.0)

    parser.add_argument("--init-seed", type=int, default=2023)
    parser.add_argument("--mask-seed", type=int, default=2024)
    parser.add_argument("--dropout-seed", type=int, default=2025)

    parser.add_argument("--input-features", type=int, default=225)
    parser.add_argument("--input-length", type=int, default=512)
    parser.add_argument("--label-length", type=int, default=64)
    parser.add_argument("--average-layers", type=int, default=6)

    parser.add_argument("--num-masks", type=int, default=8)
    parser.add_argument("--mask-block", type=float, default=5)
    parser.add_argument("--mask-prob", type=float, default=0.5)
    parser.add_argument("--mask-adjust", type=float, default=0.05)
    parser.add_argument("--mask-stdev", type=float, default=0.01)

    parser.add_argument("--ema-start", type=float, default=0.999)
    parser.add_argument("--ema-end", type=float, default=0.99999)
    parser.add_argument("--ema-steps", type=int, default=100000)

    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--adam-beta1", type=float, default=0.9)
    parser.add_argument("--adam-beta2", type=float, default=0.98)
    parser.add_argument("--adam-eps", type=float, default=1e-6)
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)

    parser.add_argument("--name", default="transformer-6l-256d")
    parser.add_argument("--project", default="gaslfr-data2vec2-pretraining")
    parser.add_argument("--warmup-steps", type=int, default=10000)
    parser.add_argument("--training-steps", type=int, default=200000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--ckpt-interval", type=int, default=10000)
    parser.add_argument("--num-keep-ckpt", type=int, default=3)
    main(parser.parse_args())
