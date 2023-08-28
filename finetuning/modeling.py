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
from functools import partial
from typing import Any

import flax.linen as nn
import flax.linen.initializers as init
import jax
import jax.numpy as jnp
from chex import Array

DenseGeneral = partial(nn.DenseGeneral, kernel_init=init.normal(0.02))
Dense = partial(nn.Dense, kernel_init=init.normal(0.02))
Embed = partial(nn.Embed, embedding_init=init.normal(0.02))
Conv = partial(nn.Conv, kernel_init=init.normal(0.02))


@dataclass
class TransformerBase:
    layers: int
    dim: int
    heads: int
    labels: int
    dropout: float = 0.1
    layerdrop: float = 0.1
    use_lstm_head: bool = False

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(TransformerBase)}


class LayerDrop(nn.Module):
    rate: float

    def sample_layerdrop_mask(self, x: Array) -> Array:
        mask_shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return jax.random.uniform(self.make_rng("dropout"), mask_shape) < self.rate

    def __call__(self, x: Array, z: Array, det: bool = True) -> Array:
        if det or self.rate == 0.0:
            return z
        return jnp.where(self.sample_layerdrop_mask(x), x, z)


class Attention(TransformerBase, nn.Module):
    def setup(self):
        self.wq = DenseGeneral((self.heads, self.head_dim))
        self.wk = DenseGeneral((self.heads, self.head_dim))
        self.wv = DenseGeneral((self.heads, self.head_dim))
        self.wo = DenseGeneral(self.dim, axis=(-2, -1))
        self.drop = nn.Dropout(self.dropout)

    def apply_rotary_embedding(self, x: Array, pos: Array) -> Array:
        freqs = 10000.0 ** -jnp.linspace(0, 1, x.shape[-1] // 2, endpoint=False)
        theta = pos[:, :, None, None] * freqs[None, None, None, :]

        cos, sin, (rx, ry) = jnp.cos(theta), jnp.sin(theta), jnp.split(x, 2, axis=-1)
        return jnp.concatenate((rx * cos - ry * sin, rx * sin + ry * cos), axis=-1)

    def __call__(self, x: Array, pos: Array, mask: Array, det: bool = True) -> Array:
        q = self.apply_rotary_embedding(self.wq(x), pos)
        k = self.apply_rotary_embedding(self.wk(x), pos)
        v = self.wv(x)

        x = jnp.einsum("bqhd,bkhd->bhqk", q, k) / k.shape[-1] ** 0.5
        x = jnp.einsum("bhqk,bkhd->bqhd", self.drop(nn.softmax(x + mask), det), v)
        return self.drop(self.wo(x), det)


class FeedForward(TransformerBase, nn.Module):
    def setup(self):
        self.w1 = Dense(self.hidden_dim)
        self.w2 = Dense(self.dim)
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, det: bool = True) -> Array:
        return self.drop(self.w2(nn.gelu(self.w1(x))), det)


class TransformerLayer(TransformerBase, nn.Module):
    def setup(self):
        self.attn = Attention(**self.kwargs)
        self.ff = FeedForward(**self.kwargs)
        self.norm_attn = nn.LayerNorm()
        self.norm_ff = nn.LayerNorm()

    def __call__(self, x: Array, pos: Array, mask: Array, det: bool = True) -> Array:
        x = x + self.attn(self.norm_attn(x), pos, mask, det)
        x = x + self.ff(self.norm_ff(x), det)
        return x


class BiLSTMHead(TransformerBase, nn.Module):
    def setup(self):
        self.lstm = nn.Bidirectional(
            forward_rnn=nn.RNN(nn.LSTMCell(self.dim)),
            backward_rnn=nn.RNN(nn.LSTMCell(self.dim)),
        )
        self.head = Dense(self.labels)

    def __call__(self, x: Array, mask: Array) -> Array:
        return self.head(self.lstm(x, seq_lengths=(mask == 0).sum((1, 2, 3))))


class Transformer(TransformerBase, nn.Module):
    def setup(self):
        self.wte = Conv(self.dim, kernel_size=(3,))
        self.layer = [TransformerLayer(**self.kwargs) for _ in range(self.layers)]
        self.norm = nn.LayerNorm()

        if self.use_lstm_head:
            self.head = BiLSTMHead(**self.kwargs)
        else:
            self.head = Dense(self.labels)

        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = LayerDrop(self.layerdrop)

    def __call__(self, x: Array, mask: Array, det: bool = True) -> tuple[Array, Array]:
        pos = jnp.arange(x.shape[1])[None, :]
        mask = -1e10 * (1 - mask[:, None, None, :].astype(jnp.float32))

        x = self.drop1(self.wte(x), det)
        for layer in self.layer:
            x = self.drop2(x, layer(x, pos, mask, det), det)
        x = self.norm(x)

        if self.use_lstm_head:
            return self.head(x, mask), x
        return self.head(x), x
