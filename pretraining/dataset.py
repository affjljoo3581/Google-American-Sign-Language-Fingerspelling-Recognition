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
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from transform import Transform, create_transform


@dataclass
class TrainDataset(Dataset):
    filenames: list[str]
    transform: Transform
    min_length: int | None = None

    def __len__(self) -> int:
        return len(self.filenames)

    def find_dominant_hand_landmarks(self, landmarks: torch.Tensor) -> torch.Tensor:
        left, right = landmarks[:, :21, :], landmarks[:, 21:42, :]
        left_valids, right_valids = (~left.isnan()).sum(), (~right.isnan()).sum()

        # Return the hands randomly when they have both enough valid frames.
        if min(left_valids, right_valids) / max(left_valids, right_valids) > 0.5:
            return left if np.random.random() < 0.5 else right
        return left if left_valids > right_valids else right

    def __getitem__(self, index: int) -> torch.Tensor:
        landmarks = torch.as_tensor(np.load(self.filenames[index]), dtype=torch.float32)
        landmarks = landmarks.unflatten(1, (-1, 3))

        hand = self.find_dominant_hand_landmarks(landmarks)
        landmarks = torch.cat((hand, landmarks[:, 42:]), dim=1)

        # Replace with another sample when number of frames is less than
        # `self.min_length` if it is specified.
        if landmarks.size(0) < (self.min_length or 0):
            return self[np.random.randint(len(self))]

        transformed = self.transform(landmarks=landmarks, label="")
        landmarks = transformed["landmarks"].nan_to_num(0).flatten(1)
        return landmarks


def create_train_dataloader(args: argparse.Namespace) -> DataLoader:
    labels = pd.read_csv(args.labels, index_col="sequence_id")
    _, vi = train_test_split(np.arange(len(labels)), test_size=0.05, random_state=42)
    valid_indices = labels.iloc[vi].index

    filenames = []
    for filename in glob.glob(args.filenames):
        if int(filename.split("/")[-1].replace(".npy", "")) not in valid_indices:
            filenames.append(filename)

    dataset = TrainDataset(
        filenames=glob.glob(args.filenames),
        transform=create_transform("default", args.input_length),
        min_length=args.min_length,
    )
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=True,
        persistent_workers=True,
    )
