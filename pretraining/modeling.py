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
    dropout: float = 0.1
    layerdrop: float = 0.1

    @property
    def head_dim(self) -> int:
        return self.dim // self.heads

    @property
    def hidden_dim(self) -> int:
        return 4 * self.dim

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(TransformerBase)}


@dataclass
class ConvStackBase:
    layers: int
    dim: int
    kernel: int
    labels: int
    dropout: float = 0.1
    layerdrop: float = 0.1

    @property
    def kwargs(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in fields(ConvStackBase)}


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


class Transformer(TransformerBase, nn.Module):
    def setup(self):
        self.wte = Conv(self.dim, kernel_size=(3,))
        self.layer = [TransformerLayer(**self.kwargs) for _ in range(self.layers)]
        self.norm = nn.LayerNorm()

        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = LayerDrop(self.layerdrop)

    def __call__(
        self, x: Array, pos: Array, mask: Array, det: bool = True
    ) -> tuple[Array, list[Array]]:
        x = self.drop1(self.wte(x), det)
        mask = -1e10 * (1 - mask[:, None, None, :].astype(jnp.float32))

        xs = []
        for layer in self.layer:
            xs.append((x := self.drop2(x, layer(x, pos, mask, det), det)))
        return self.norm(x), xs


class ConvStackLayer(ConvStackBase, nn.Module):
    def setup(self):
        self.w1 = Dense(self.dim)
        self.w2 = Conv(self.dim, (self.kernel,), feature_group_count=self.dim)
        self.w3 = Dense(self.dim)

        self.norm = nn.LayerNorm()
        self.drop = nn.Dropout(self.dropout)

    def __call__(self, x: Array, mask: Array, det: bool = True) -> Array:
        z = nn.gelu(self.w1(x))
        z = nn.gelu(self.norm(self.w2(z * mask)))
        return x + self.drop(self.w3(z), det)


class ConvStack(ConvStackBase, nn.Module):
    def setup(self):
        self.wte = Conv(self.dim, kernel_size=(3,))
        self.layer = [ConvStackLayer(**self.kwargs) for _ in range(self.layers)]
        self.head = Dense(self.labels)

        self.drop1 = nn.Dropout(self.dropout)
        self.drop2 = LayerDrop(self.layerdrop)

    def __call__(self, x: Array, mask: Array, det: bool = True) -> Array:
        x = self.drop1(self.wte(x), det)
        mask = mask[:, :, None].astype(jnp.float32)

        for layer in self.layer:
            x = self.drop2(x, layer(x, mask, det), det)
        return self.head(x)


class Data2vecTeacher(nn.Module):
    encoder: Transformer
    average_layers: int = 6

    def masked_instance_norm(self, x: Array, mask: Array) -> Array:
        mask, lengths = mask[:, :, None], jnp.maximum(mask.sum(1, keepdims=True), 1.0)
        x = x - ((x * mask).sum(1) / lengths)[:, None, :]
        x = x * jax.lax.rsqrt((x**2 * mask).sum(1) / lengths + 1e-6)[:, None, :]
        return x

    def __call__(self, x: Array) -> Array:
        pos, mask = jnp.arange(x.shape[1])[None, :], (x != -100.0).any(2)
        xs = self.encoder(jnp.where(x == -100.0, 0.0, x), pos, mask, det=True)[1]
        xs = xs[-self.average_layers :]

        x = sum(self.masked_instance_norm(x, mask) for x in xs) / len(xs)
        return x * mask[..., None]


class Data2vecStudent(nn.Module):
    encoder: Transformer
    decoder: ConvStack

    num_masks: int = 8
    mask_block: int = 5
    mask_prob: float = 0.5
    mask_adjust: float = 0.05
    mask_stdev: float = 0.01

    def create_feature_mask(self, x: Array) -> Array:
        rate = (1 - self.mask_prob + self.mask_adjust) / self.mask_block
        mask = jax.random.bernoulli(self.make_rng("mask"), rate, x.shape[:2])
        return nn.max_pool(mask[..., None], (self.mask_block,), padding="SAME")[..., 0]

    def __call__(self, x: Array) -> tuple[Array, Array]:
        x = jnp.repeat(x, repeats=self.num_masks, axis=0)
        input_length = int(x.shape[1] * (1 - self.mask_prob))

        padding_mask, feature_mask = (x != -100.0).any(2), self.create_feature_mask(x)

        label_mask = padding_mask & ~feature_mask
        input_mask = padding_mask & feature_mask
        input_mask = input_mask & (jnp.cumsum(input_mask, axis=1) <= input_length)

        # Prepare the indices for gathering and updating sparse tokens. We will use only
        # `input_length` tokens, thus gathering indices should be truncated and
        # insertion indices should be less than `input_length`.
        get_indices = jnp.argsort(~input_mask, axis=1)
        put_indices = jnp.argsort(get_indices, axis=1)

        get_indices = get_indices[:, :input_length]
        put_indices = jnp.minimum(put_indices, input_length - 1)

        # Get feature representations of the sparse tokens and restore the features to
        # their original positions. The masked tokens  will be replaced with noises.
        x = jnp.where(x == -100.0, 0.0, x)
        x, _ = self.encoder(
            x=jnp.take_along_axis(x, get_indices[..., None], axis=1),
            pos=get_indices,
            mask=jnp.take_along_axis(input_mask, get_indices, axis=1),
            det=False,
        )
        x = jnp.take_along_axis(x, put_indices[..., None], axis=1)

        noise = self.mask_stdev * jax.random.normal(self.make_rng("mask"), x.shape)
        x = jnp.where(input_mask[..., None], x, noise)
        return self.decoder(x, padding_mask, det=False), label_mask
