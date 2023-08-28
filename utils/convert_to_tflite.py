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
import zipfile
from collections.abc import Generator
from functools import partial
from typing import Callable

import flax
import numpy as np
import pandas as pd
import tensorflow as tf
from jax.experimental import jax2tf
from mediapipe.python.solutions.face_mesh import FACEMESH_LIPS
from sklearn.model_selection import train_test_split

from modeling import Transformer

LEFT_HAND_COLUMNS = [
    f"{axis}_left_hand_{index}" for index in range(21) for axis in "xyz"
]
RIGHT_HAND_COLUMNS = [
    f"{axis}_right_hand_{index}" for index in range(21) for axis in "xyz"
]
ARMS_COLUMNS = [f"{axis}_pose_{index}" for index in range(11, 25) for axis in "xyz"]
LIPS_COLUMNS = [
    f"{axis}_face_{index}"
    for index in sorted(set(sum(map(list, FACEMESH_LIPS), [])))
    for axis in "xyz"
]
INPUT_COLUMNS = LEFT_HAND_COLUMNS + RIGHT_HAND_COLUMNS + ARMS_COLUMNS + LIPS_COLUMNS

INPUT_SIGNATURE = tf.TensorSpec(
    shape=[None, len(INPUT_COLUMNS)], dtype=tf.float32, name="inputs"
)
DEFAULT_INPUT_SCALES = (1.0, 1.8979039030629028, 1.0)


def create_representative_dataset(
    args: argparse.Namespace,
) -> Callable[[], Generator[dict[str, np.ndarray]]]:
    labels = pd.read_csv(args.labels, index_col="sequence_id")
    _, idx = train_test_split(np.arange(len(labels)), test_size=0.05, random_state=42)

    def representative_dataset() -> Generator[dict[str, np.ndarray]]:
        for sample in labels.iloc[idx].sample(500, random_state=42).itertuples():
            landmarks = np.load(args.filepath.format(sample.Index))
            landmarks = landmarks.reshape(-1, 96, 3) / DEFAULT_INPUT_SCALES
            landmarks = landmarks.reshape(-1, 288).astype(np.float32)
            yield {"inputs": landmarks}

    return representative_dataset


class TFLiteModule(tf.Module):
    def __init__(self, model_fn: tf.function, return_logits: bool = False):
        super().__init__()
        self.model_fn = model_fn
        self.return_logits = return_logits

    @tf.function(jit_compile=True)
    def normalize_landmarks(self, landmarks: tf.Tensor) -> tf.Tensor:
        landmarks = landmarks * tf.constant(DEFAULT_INPUT_SCALES)

        x_mean = tf.reduce_mean(landmarks[:, :, 0][~tf.math.is_nan(landmarks[:, :, 0])])
        y_mean = tf.reduce_mean(landmarks[:, :, 1][~tf.math.is_nan(landmarks[:, :, 1])])
        z_mean = tf.reduce_mean(landmarks[:, :, 2][~tf.math.is_nan(landmarks[:, :, 2])])
        landmarks = landmarks - tf.stack((x_mean, y_mean, z_mean))

        x_abs_max = landmarks[:, :, 0][~tf.math.is_nan(landmarks[:, :, 0])]
        x_abs_max = tf.reduce_max(tf.abs(x_abs_max))
        y_abs_max = landmarks[:, :, 1][~tf.math.is_nan(landmarks[:, :, 1])]
        y_abs_max = tf.reduce_max(tf.abs(y_abs_max))
        return landmarks / tf.maximum(x_abs_max, y_abs_max)

    @tf.function(jit_compile=True)
    def preprocess(self, landmarks: tf.Tensor) -> tf.Tensor:
        landmarks = tf.reshape(landmarks, (-1, 96, 3))
        left_hand, right_hand = landmarks[:, :21, :], landmarks[:, 21:42, :]

        num_left_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(left_hand), tf.float32))
        num_right_nans = tf.reduce_sum(tf.cast(tf.math.is_nan(right_hand), tf.float32))

        hand = left_hand if num_left_nans < num_right_nans else right_hand
        arms, lips = landmarks[:, 42:56], landmarks[:, 56:96]

        hand = self.normalize_landmarks(hand)
        arms = self.normalize_landmarks(arms)
        lips = self.normalize_landmarks(lips)
        landmarks = tf.reshape(tf.concat((hand, arms, lips), axis=1), (-1, 225))
        landmarks = tf.where(tf.math.is_nan(landmarks), 0.0, landmarks)
        return landmarks

    @tf.function(jit_compile=True)
    def decode_ctc_logits(self, logits: tf.Tensor) -> tf.Tensor:
        preds = tf.argmax(logits, axis=1, output_type=tf.int32)
        dedups = tf.where(preds[:-1] == preds[1:], logits.shape[-1] - 1, preds[1:])
        return tf.one_hot(tf.concat((preds[0][None], dedups), axis=0), logits.shape[-1])

    @tf.function(input_signature=[INPUT_SIGNATURE], jit_compile=True)
    def __call__(self, inputs: tf.Tensor) -> dict[str, tf.Tensor]:
        landmarks = self.preprocess(inputs)

        mask = tf.reduce_any(landmarks[:, :63] != 0.0, axis=1)
        if tf.reduce_sum(tf.cast(mask, tf.int32)) == 0:
            return {"outputs": tf.zeros((1, 1))}

        logits = self.model_fn(landmarks[None])[0]
        if self.return_logits:
            return {"outputs": logits}
        return {"outputs": self.decode_ctc_logits(logits)}


def main(args: argparse.Namespace):
    with open(args.vocab) as fp:
        vocab = json.load(fp)
    with open(args.checkpoint, "rb") as fp:
        params = flax.serialization.msgpack_restore(fp.read())["model"]

    model = Transformer(
        layers=len([x for x in params if x.startswith("layer_")]),
        dim=params["wte"]["kernel"].shape[-1],
        heads=params["layer_0"]["attn"]["wq"]["kernel"].shape[-2],
        labels=len(vocab) + 1,
    )
    model_fn = jax2tf.convert(
        partial(model.apply, {"params": params}),
        polymorphic_shapes=[jax2tf.PolyShape(1, "T", 225)],
        enable_xla=False,
    )

    module = TFLiteModule(model_fn, args.return_logits)
    converter = tf.lite.TFLiteConverter.from_concrete_functions(
        [module.__call__.get_concrete_function()], module
    )
    if args.quant_type == "float16":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif args.quant_type == "int8":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    if args.quantize_act:
        converter.representative_dataset = create_representative_dataset(args)

    if args.tflite_only:
        with open(args.checkpoint + ".tflite", "wb") as fp:
            fp.write(converter.convert())
    else:
        with zipfile.ZipFile(args.checkpoint + "-submission.zipp", "w") as zfp:
            inference_args = json.dumps({"selected_columns": INPUT_COLUMNS})
            zfp.writestr("inference_args.json", inference_args)
            zfp.writestr("model.tflite", converter.convert())


if __name__ == "__main__":
    DEFAULT_FILEPATH = "resources/competition/train_landmarks_npy/{}.npy"
    DEFAULT_LABEL_PATH = "resources/competition/train.csv"
    DEFAULT_VOCAB_PATH = "resources/competition/character_to_prediction_index.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--filepath", default=DEFAULT_FILEPATH)
    parser.add_argument("--labels", default=DEFAULT_LABEL_PATH)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--quant-type", default="float16")
    parser.add_argument("--return-logits", action="store_true", default=False)
    parser.add_argument("--tflite-only", action="store_true", default=False)
    parser.add_argument("--quantize-act", action="store_true", default=False)
    main(parser.parse_args())
