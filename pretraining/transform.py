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

from abc import ABC, abstractmethod
from typing import Any, Callable

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial.transform import Rotation


class Transform(ABC):
    def __init__(self, *, p: float | None = None):
        self.p = p

    @abstractmethod
    def apply(self, **inputs: Any) -> dict[str, Any]:
        pass

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        if self.p is None or np.random.random() < self.p:
            return self.apply(**inputs)
        return inputs


class Sequential(Transform):
    def __init__(self, *transforms: Callable, **kwargs: Any):
        super().__init__(**kwargs)
        self.transforms = transforms

    def apply(self, **inputs: Any) -> dict[str, Any]:
        for transform in self.transforms:
            inputs = transform(**inputs)
        return inputs


class LandmarkGroups(Transform):
    def __init__(self, transform: Transform, lengths: list[int], **kwargs: Any):
        super().__init__(**kwargs)
        self.transform = transform
        self.lengths = lengths

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        outputs = []
        for length in self.lengths:
            outputs.append(self.transform(landmarks=landmarks[:, :length], **inputs))
            landmarks = landmarks[:, length:]
        landmarks = torch.cat([x["landmarks"] for x in outputs], dim=1)
        return dict(outputs[0], landmarks=landmarks)


class Normalize(Transform):
    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        landmarks = landmarks - landmarks.nanmean((0, 1))
        landmarks = landmarks / landmarks.abs().nan_to_num(0)[:, :, :2].amax()
        return dict(inputs, landmarks=landmarks)


class Truncate(Transform):
    def __init__(self, length: int, **kwargs: Any):
        super().__init__(**kwargs)
        self.length = length

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        return dict(inputs, landmarks=landmarks[: self.length])


class Pad(Transform):
    def __init__(self, length: int, value: float = -100.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.length = length
        self.value = value

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        if (padding := self.length - landmarks.size(0)) > 0:
            landmarks = F.pad(landmarks, (0, 0, 0, 0, 0, padding), value=self.value)
        return dict(inputs, landmarks=landmarks)


class AlignCTCLabel(Transform):
    def apply(
        self, landmarks: torch.Tensor, label: str, **inputs: Any
    ) -> dict[str, Any]:
        aligned = "".join(f"▁{j}" if i == j else j for i, j in zip(f"▁{label}", label))
        return dict(
            inputs,
            landmarks=landmarks,
            label=label,
            ctc_label=aligned[: landmarks.size(0)],
        )


class HorizontalFlip(Transform):
    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        landmarks = landmarks * torch.tensor([-1.0, 1.0, 1.0])
        return dict(inputs, landmarks=landmarks)


class TimeFlip(Transform):
    def apply(self, **inputs: Any) -> dict[str, Any]:
        inputs["landmarks"] = inputs["landmarks"].flip(0)
        for name in inputs:
            if "label" in name:
                inputs[name] = inputs[name][::-1]
        return inputs


class RandomResample(Transform):
    def __init__(self, limit: float = 0.2, **kwargs: Any):
        super().__init__(**kwargs)
        self.limit = limit

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        # First of all, we need to fill all `nan` values from their previous frame
        # because common linear interpolation uses the previous and next frames to
        # literally interpolate the intermediate values. It makes other frames be
        # invalid and hence we should fill them with proper values.
        x_ff = landmarks.clone()
        for i in range(1, x_ff.size(0)):
            x_ff[i].copy_(x_ff[i].where(~landmarks[i].isnan(), x_ff[i - 1]))

        x_ff = x_ff.flatten(1).transpose(1, 0).unsqueeze(0)
        mask = (~landmarks.isnan()).flatten(1).transpose(1, 0).unsqueeze(0).float()

        # After that, we interpolate the frames as well as their valid mask to infer
        # which positions should be filled with `nan` for the interpolated video.
        scale = 1 + np.random.uniform(-self.limit, self.limit)
        x_ff = F.interpolate(x_ff, scale_factor=scale, mode="linear")[0].T
        mask = F.interpolate(mask, scale_factor=scale, mode="linear")[0].T

        x_ff.masked_fill_(mask < 0.5, torch.nan)
        return dict(inputs, landmarks=x_ff.unflatten(1, (-1, 3)))


class RandomShift(Transform):
    def __init__(self, stdev: float = 0.05, **kwargs: Any):
        super().__init__(**kwargs)
        self.stdev = stdev

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        landmarks = landmarks + torch.empty(3).normal_(0, self.stdev)
        return dict(inputs, landmarks=landmarks)


class RandomScale(Transform):
    def __init__(self, limit: float = 0.2, **kwargs: Any):
        super().__init__(**kwargs)
        self.limit = limit

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        landmarks = landmarks * (1 + torch.empty(3).uniform_(-self.limit, self.limit))
        return dict(inputs, landmarks=landmarks)


class RandomShear(Transform):
    def __init__(self, limit: float = 0.2, **kwargs: Any):
        super().__init__(**kwargs)
        self.limit = limit

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        axis = np.random.choice(3)
        rest = list(set(range(3)) - {axis})

        S = torch.eye(3)
        S[rest, axis] = torch.empty(2).uniform_(-self.limit, self.limit)
        return dict(inputs, landmarks=torch.einsum("ij,bni->bnj", S, landmarks))


class RandomInterpolatedRotation(Transform):
    def __init__(
        self,
        center_stdev: float = 0.5,
        angle_limit: float = np.pi / 4,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.center_stdev = center_stdev
        self.angle_limit = angle_limit

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        # In this rotation augmentation, the rotation center point and its rotation
        # angles will be contiguously changed throughout the timeline.
        offset = torch.lerp(
            torch.empty(3).normal_(0, self.center_stdev)[None, :],
            torch.empty(3).normal_(0, self.center_stdev)[None, :],
            torch.linspace(0, 1, landmarks.size(0))[:, None],
        )
        rotvec = torch.lerp(
            torch.empty(3).uniform_(-self.angle_limit, self.angle_limit)[None, :],
            torch.empty(3).uniform_(-self.angle_limit, self.angle_limit)[None, :],
            torch.linspace(0, 1, landmarks.size(0))[:, None],
        )

        R = Rotation.from_rotvec(rotvec.numpy()).as_matrix()
        R = torch.as_tensor(R, dtype=torch.float32)

        # To rotate the landmarks at a random point while maintaining original zero
        # point, we begin with subtracting the rotation center. Subsequently, a random
        # rotation is applied and the landmarks are then restored by adding the center.
        landmarks = landmarks - offset[:, None, :]
        landmarks = torch.einsum("bij,bni->bnj", R, landmarks) + offset[:, None, :]
        return dict(inputs, landmarks=landmarks)


class FrameBlockMask(Transform):
    def __init__(self, ratio: float = 0.1, block_size: int = 3, **kwargs: Any):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.block_size = block_size

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        # Because frames around the anchors will be masked as well, the probability
        # should be divided by the block size to maintain the masking ratio.
        mask = ~landmarks.isnan().all(2).all(1)
        mask = mask & (torch.rand(landmarks.size(0)) < self.ratio / self.block_size)
        mask = F.max_pool1d(mask[None].float(), self.block_size, 1, 1).squeeze(0).bool()

        landmarks = landmarks.masked_fill(mask[:, None, None], torch.nan)
        return dict(inputs, landmarks=landmarks)


class FrameNoise(Transform):
    def __init__(self, ratio: float = 0.1, noise_stdev: float = 1.0, **kwargs: Any):
        super().__init__(**kwargs)
        self.ratio = ratio
        self.noise_stdev = noise_stdev

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        mask = ~landmarks.isnan().all(2).all(1)
        mask = mask & (torch.rand(landmarks.size(0)) < self.ratio)
        noise = landmarks.clone().normal_(0, self.noise_stdev)
        return dict(inputs, landmarks=landmarks.where(~mask[:, None, None], noise))


class FeatureMask(Transform):
    def __init__(self, ratio: float = 0.1, **kwargs: Any):
        super().__init__(**kwargs)
        self.ratio = ratio

    def apply(self, landmarks: torch.Tensor, **inputs: Any) -> dict[str, Any]:
        mask = torch.rand(landmarks.size(1)) < self.ratio
        landmarks = landmarks.masked_fill(mask[None, :, None], torch.nan)
        return dict(inputs, landmarks=landmarks)


class AlignedFrameRoll(Transform):
    def __init__(self, ratio: float = 0.5, **kwargs: Any):
        super().__init__(**kwargs)
        self.ratio = ratio

    def apply(
        self, landmarks: torch.Tensor, aligned_label: str, **inputs: Any
    ) -> dict[str, Any]:
        roll = np.random.uniform(-self.ratio, self.ratio)
        landmarks = landmarks.roll(int(roll * landmarks.size(0)), dim=0)

        roll = int(roll * len(aligned_label))
        aligned_label = aligned_label[-roll:] + aligned_label[:-roll]
        return dict(inputs, landmarks=landmarks, aligned_label=aligned_label)


class CutMix(Transform):
    def __init__(self, transform: Transform, blank_token: str = "▁", **kwargs: Any):
        super().__init__(**kwargs)
        self.transform = transform
        self.blank_token = blank_token

    def merge_label_alignments(self, label: str) -> str:
        label = label[0] + "".join(j for i, j in zip(label, label[1:]) if i != j)
        label = label.replace(self.blank_token, "")
        return label

    def __call__(self, **inputs: Any) -> dict[str, Any]:
        outputs = super().__call__(**inputs)
        if "org" in outputs:
            outputs = self.transform(**inputs["org"])
            return outputs
        return outputs

    def apply(self, org: dict[str, Any], ref: dict[str, Any]) -> dict[str, Any]:
        org_output, ref_output = self.transform(**org), self.transform(**ref)
        org_landmarks, org_label = org_output["landmarks"], org_output["aligned_label"]
        ref_landmarks, ref_label = ref_output["landmarks"], ref_output["aligned_label"]

        for _ in range(100):
            length = min(org_landmarks.size(0), ref_landmarks.size(0))
            length = np.random.randint(length)
            src = np.random.randint(ref_landmarks.size(0) - length)
            dst = np.random.randint(org_landmarks.size(0) - length)

            src_label_length = int(length / ref_landmarks.size(0) * len(ref_label))
            dst_label_length = int(length / org_landmarks.size(0) * len(org_label))
            src_label = int(src / ref_landmarks.size(0) * len(ref_label))
            dst_label = int(dst / org_landmarks.size(0) * len(org_label))

            label = org_label[:dst_label]
            label += ref_label[src_label : src_label + src_label_length]
            label += org_label[dst_label + dst_label_length :]

            if label := self.merge_label_alignments(label):
                break

        if not label:
            return org_output

        org_landmarks[dst : dst + length] = ref_landmarks[src : src + length]
        return {"landmarks": org_landmarks, "label": label}


def create_transform(augmentation: str, max_length: int) -> Transform:
    if augmentation == "none":
        return Sequential(
            LandmarkGroups(Normalize(), lengths=(21, 14, 40)),
            Truncate(max_length),
            AlignCTCLabel(),
            Pad(max_length),
        )
    elif augmentation == "default":
        return Sequential(
            LandmarkGroups(Normalize(), lengths=(21, 14, 40)),
            TimeFlip(p=0.5),
            RandomResample(limit=0.5, p=0.5),
            Truncate(max_length),
            AlignCTCLabel(),
            FrameBlockMask(ratio=0.1, block_size=3, p=0.25),
            FrameNoise(ratio=0.1, noise_stdev=0.3, p=0.25),
            FeatureMask(ratio=0.1, p=0.1),
            LandmarkGroups(
                Sequential(
                    HorizontalFlip(p=0.5),
                    RandomInterpolatedRotation(0.2, np.pi / 4, p=0.5),
                    RandomShear(limit=0.2),
                    RandomScale(limit=0.2),
                    RandomShift(stdev=0.1),
                ),
                lengths=(21, 14, 40),
            ),
            Pad(max_length),
        )
    elif augmentation == "cutmix":
        return Sequential(
            CutMix(
                Sequential(
                    LandmarkGroups(Normalize(), lengths=(21, 14, 40)),
                    TimeFlip(p=0.5),
                    RandomResample(limit=0.5, p=0.5),
                    FrameBlockMask(ratio=0.1, block_size=3, p=0.25),
                    FrameNoise(ratio=0.1, noise_stdev=0.3, p=0.25),
                    FeatureMask(ratio=0.1, p=0.1),
                    LandmarkGroups(
                        Sequential(
                            HorizontalFlip(p=0.5),
                            RandomInterpolatedRotation(0.2, np.pi / 4, p=0.5),
                            RandomShear(limit=0.2),
                            RandomScale(limit=0.2),
                            RandomShift(stdev=0.1),
                        ),
                        lengths=(21, 14, 40),
                    ),
                ),
                blank_token="▁",
            ),
            Truncate(max_length),
            AlignCTCLabel(),
            Pad(max_length),
        )
