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

from dataclasses import dataclass, fields
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
from chex import Array


@dataclass
class TransformerBase:
    layers: int
    dim: int
    heads: int
    labels: int

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(TransformerBase)}


class Attention(TransformerBase, nn.Module):
    def setup(self):
        self.wq = nn.DenseGeneral((self.heads, self.head_dim))
        self.wk = nn.DenseGeneral((self.heads, self.head_dim))
        self.wv = nn.DenseGeneral((self.heads, self.head_dim))
        self.wo = nn.DenseGeneral(self.dim, axis=(-2, -1))

    def apply_rotary_embedding(self, x: Array) -> Array:
        freqs = 10000.0 ** -jnp.linspace(0, 1, (half_dim := x.shape[-1] // 2), False)
        theta = jnp.arange(x.shape[1], dtype=jnp.float32)
        theta = theta[None, :, None, None] * freqs[None, None, None, :]

        cos, sin = jnp.cos(theta), jnp.sin(theta)
        rx = jax.lax.dynamic_slice_in_dim(x, 0, half_dim, axis=-1)
        ry = jax.lax.dynamic_slice_in_dim(x, half_dim, half_dim, axis=-1)
        return jnp.concatenate((rx * cos - ry * sin, rx * sin + ry * cos), axis=-1)

    def __call__(self, x: Array) -> Array:
        q = self.apply_rotary_embedding(self.wq(x))
        k = self.apply_rotary_embedding(self.wk(x))
        v = self.wv(x)

        x = jnp.einsum("bqhd,bkhd->bhqk", q, k) / k.shape[-1] ** 0.5
        x = jnp.einsum("bhqk,bkhd->bqhd", nn.softmax(x), v)
        return self.wo(x)


class FeedForward(TransformerBase, nn.Module):
    def setup(self):
        self.w1 = nn.Dense(self.hidden_dim)
        self.w2 = nn.Dense(self.dim)

    def __call__(self, x: Array) -> Array:
        return self.w2(nn.gelu(self.w1(x)))


class TransformerLayer(TransformerBase, nn.Module):
    def setup(self):
        self.attn = Attention(**self.kwargs)
        self.ff = FeedForward(**self.kwargs)
        self.norm_attn = nn.LayerNorm(use_fast_variance=False)
        self.norm_ff = nn.LayerNorm(use_fast_variance=False)

    def __call__(self, x: Array) -> Array:
        x = x + self.attn(self.norm_attn(x))
        x = x + self.ff(self.norm_ff(x))
        return x


class Transformer(TransformerBase, nn.Module):
    def setup(self):
        self.wte = nn.Conv(self.dim, kernel_size=(3,))
        self.layer = [TransformerLayer(**self.kwargs) for _ in range(self.layers)]
        self.norm = nn.LayerNorm(use_fast_variance=False)
        self.head = nn.Dense(self.labels)

    def __call__(self, x: Array) -> Array:
        x = self.wte(x)
        for layer in self.layer:
            x = layer(x)
        return self.head(self.norm(x))
