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
import json
import os
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from transform import Transform, create_transform


@dataclass
class TrainDataset(Dataset):
    filenames: list[str]
    labels: pd.DataFrame
    alignments: pd.DataFrame
    vocab: dict[str, int]
    transform: Transform
    label_length: int = 64
    use_cutmix: bool = False
    len_ratio_filter: float | None = None

    def __len__(self) -> int:
        return len(self.filenames)

    def find_dominant_hand_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        left, right = landmarks[:, :21, :], landmarks[:, 21:42, :]
        return left if left.isnan().sum() < right.isnan().sum() else right

    def prepare_sample(self, index: int) -> dict[str, Any]:
        seq_id = int(os.path.basename(self.filenames[index]).replace(".npy", ""))
        phrase = self.labels.loc[seq_id].phrase
        alignment = self.alignments.loc[seq_id].alignment

        landmarks = torch.as_tensor(np.load(self.filenames[index]), dtype=torch.float32)
        landmarks = landmarks.unflatten(1, (-1, 3))

        hand = self.find_dominant_hand_landmarks(landmarks)
        landmarks = torch.cat((hand, landmarks[:, 42:]), dim=1)

        # Replace with another sample when a ratio of phrase length to frame length is
        # smaller than `self.len_ratio_filter` if it is specified.
        if landmarks.size(0) / len(phrase) < (self.len_ratio_filter or 0):
            return self.prepare_sample(np.random.randint(len(self)))
        return {"landmarks": landmarks, "label": phrase, "aligned_label": alignment}

    def create_lm_labels(self, label: str) -> torch.Tensor:
        labels = [len(self.vocab)] + [self.vocab[i] for i in label] + [len(self.vocab)]
        labels = labels + [-100] * (self.label_length - len(labels))
        return torch.tensor(labels[: self.label_length])

    def create_ctc_labels(self, label: str) -> torch.Tensor:
        labels = [self.vocab[i] for i in label if i in self.vocab]
        labels = labels + [-100] * (self.label_length - len(labels))
        return torch.tensor(labels[: self.label_length])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.use_cutmix:
            transformed = self.transform(
                org=self.prepare_sample(index),
                ref=self.prepare_sample(np.random.randint(len(self))),
            )
            phrase = transformed["label"]
        else:
            inputs = self.prepare_sample(index)
            transformed = self.transform(**inputs)
            phrase = inputs["label"]

        return {
            "landmarks": transformed["landmarks"].nan_to_num(0).flatten(1),
            "lm_labels": self.create_lm_labels(transformed["label"]),
            "ctc_labels": self.create_ctc_labels(transformed["ctc_label"]),
            "phrase": phrase,
        }


def create_train_valid_dataloaders(
    args: argparse.Namespace,
) -> tuple[DataLoader, DataLoader, dict[str, int]]:
    with open(args.vocab) as fp:
        vocab = json.load(fp)

    train_labels = pd.read_csv(args.train_labels, index_col="sequence_id")
    supp_labels = pd.read_csv(args.supp_labels, index_col="sequence_id")
    alignments = pd.read_csv(args.alignments, index_col="sequence_id")

    train_labels["filename"] = train_labels.index.map(args.train_filepath.format)
    supp_labels["filename"] = supp_labels.index.map(args.supp_filepath.format)
    supp_labels = supp_labels[supp_labels.filename.map(os.path.exists)]

    ti, vi = train_test_split(range(len(train_labels)), test_size=0.05, random_state=42)
    train_filenames = train_labels.iloc[ti].filename.tolist()
    valid_filenames = train_labels.iloc[vi].filename.tolist()

    labels = pd.concat((train_labels, supp_labels))
    if args.use_supplemental:
        train_filenames.extend(supp_labels.filename.tolist())

    transform_type = "cutmix" if args.use_cutmix else "default"
    train_dataset = TrainDataset(
        filenames=train_filenames,
        labels=labels,
        alignments=alignments,
        vocab=vocab,
        transform=create_transform(transform_type, args.input_length),
        label_length=args.label_length,
        use_cutmix=args.use_cutmix,
        len_ratio_filter=1.0,
    )
    valid_dataset = TrainDataset(
        filenames=valid_filenames,
        labels=labels,
        alignments=alignments,
        vocab=vocab,
        transform=create_transform("none", args.input_length),
        label_length=args.label_length,
        use_cutmix=False,
        len_ratio_filter=None,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=True,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=True,
    )
    return train_dataloader, valid_dataloader, vocab
