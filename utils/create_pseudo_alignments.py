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
import multiprocessing as mp
import os

import numpy as np
import pandas as pd
import tflite_runtime.interpreter as tflite
import tqdm
from scipy.special import log_softmax

DEFAULT_INPUT_SCALES = np.array((1.0, 1.8979039030629028, 1.0), dtype=np.float32)


def match_ctc_label_alignment(
    logprobs: np.ndarray, label: str, vocab: dict[str, int]
) -> str:
    idx2char = {v: k for k, v in vocab.items()}
    label = np.array(sum([[len(vocab), vocab[i]] for i in label], []) + [len(vocab)])

    total_logprobs = np.full((logprobs.shape[0], label.shape[0]), fill_value=-10000.0)
    total_logprobs[0, :2] = logprobs[0, :2]
    parent_table = np.full_like(total_logprobs, fill_value=-1, dtype=np.int32)

    transitions = np.arange(label.shape[0])
    transitions = np.stack((transitions - 2, transitions - 1, transitions), axis=1)
    transitions = np.maximum(transitions, 0)

    transition_mask = label[transitions[:, 0]] == label[transitions[:, 2]]
    transitions[:, 0] = np.where(transition_mask, transitions[:, 2], transitions[:, 0])

    for t in range(1, logprobs.shape[0]):
        prev_scores = total_logprobs[t - 1][transitions]
        best_parents = np.argmax(prev_scores, axis=1)
        best_parents = np.take_along_axis(transitions, best_parents[:, None], axis=1)

        total_logprobs[t] = logprobs[t][label] + np.max(prev_scores, axis=1)
        parent_table[t] = best_parents.flatten()

    finish_token = (
        label.shape[0] - 1
        if total_logprobs[-1, -1] > total_logprobs[-1, -2]
        else label.shape[0] - 2
    )
    best_path = [finish_token]
    for t in range(parent_table.shape[0] - 1, 0, -1):
        best_path.append(parent_table[t, best_path[-1]])

    best_path = [label[i] for i in reversed(best_path)]
    return "".join(idx2char.get(i, "▁") for i in best_path)


def process_fn(args: argparse.Namespace, samples: pd.DataFrame, queue: mp.Queue):
    with open(args.vocab) as fp:
        vocab = json.load(fp)
    interpreter = tflite.Interpreter(model_path=args.model)
    predict_fn = interpreter.get_signature_runner("serving_default")

    for sample in samples.itertuples():
        if os.path.exists((filename := args.filepath.format(sample.Index))):
            landmarks = np.load(filename).astype(np.float32)
            landmarks = landmarks.reshape(-1, 96, 3) / DEFAULT_INPUT_SCALES
            landmarks = landmarks.reshape(-1, 288)

            logprobs = log_softmax(predict_fn(inputs=landmarks)["outputs"], axis=-1)
            aligned_label = match_ctc_label_alignment(logprobs, sample.phrase, vocab)
            queue.put({"sequence_id": sample.Index, "alignment": aligned_label})
        else:
            queue.put({"sequence_id": sample.Index, "alignment": ""})


def main(args: argparse.Namespace):
    labels = pd.read_csv(args.labels, index_col="sequence_id")

    queue = mp.Queue()
    for i in range(args.num_workers):
        p = mp.Process(
            target=process_fn,
            args=(args, labels.iloc[i :: args.num_workers], queue),
            daemon=True,
        )
        p.start()

    outputs = [queue.get() for _ in tqdm.trange(len(labels))]
    pd.DataFrame(outputs).to_csv("alignments.csv", index=False)


if __name__ == "__main__":
    DEFAULT_FILEPATH = "resources/competition/train_landmarks_npy/{}.npy"
    DEFAULT_LABEL_PATH = "resources/competition/train.csv"
    DEFAULT_VOCAB_PATH = "resources/competition/character_to_prediction_index.json"

    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--filepath", default=DEFAULT_FILEPATH)
    parser.add_argument("--labels", default=DEFAULT_LABEL_PATH)
    parser.add_argument("--vocab", default=DEFAULT_VOCAB_PATH)
    parser.add_argument("--num-workers", type=int, default=os.cpu_count())
    main(parser.parse_args())
