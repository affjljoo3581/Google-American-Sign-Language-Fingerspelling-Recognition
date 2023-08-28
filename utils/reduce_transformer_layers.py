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

import flax
import numpy as np


def main(args: argparse.Namespace):
    with open(args.checkpoint, "rb") as fp:
        params = flax.serialization.msgpack_restore(fp.read())
        if "student" in params:
            params = {"model": params["student"]["encoder"]}
        params = params["model"]

    total_layers = len([name for name in params if name.startswith("layer_")])
    layer_indices = (total_layers / args.layers) * np.arange(1, args.layers + 1) - 1
    layer_indices = layer_indices.astype(np.int32)

    new_params = {k: v for k, v in params.items() if not k.startswith("layer_")}
    for i, j in enumerate(layer_indices):
        new_params[f"layer_{i}"] = params[f"layer_{j}"]

    with open(f"{args.checkpoint}-layer-reduction-{args.layers}", "wb") as fp:
        fp.write(flax.serialization.msgpack_serialize({"model": new_params}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint")
    parser.add_argument("--layers", type=int, default=6)
    main(parser.parse_args())
