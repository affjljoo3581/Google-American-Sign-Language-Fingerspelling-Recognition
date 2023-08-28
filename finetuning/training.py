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
from functools import partial

import flax
import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key

from modeling import Transformer


class TrainModule(nn.Module):
    model: Transformer
    teacher: Transformer | None
    blank_id: int
    hard_distill: bool = False

    def decode_ctc_logits(self, logits: Array, mask: Array) -> Array:
        preds = jnp.argmax(logits, axis=2)
        dedup = jnp.where(preds[:, :-1] == preds[:, 1:], self.blank_id, preds[:, 1:])
        return jnp.where(mask, jnp.concatenate((preds[:, :1], dedup), 1), self.blank_id)

    @nn.compact
    def __call__(
        self, landmarks: Array, lm_labels: Array, ctc_labels: Array, det: bool = True
    ) -> tuple[ArrayTree, ArrayTree]:
        logits, hidden = self.model(
            x=(x := jnp.where(landmarks == -100.0, 0.0, landmarks)),
            mask=(mask := (landmarks != -100.0).any(2)),
            det=det,
        )
        label_mask = ctc_labels != -100

        loss = optax.ctc_loss(logits, ~mask, ctc_labels, ~label_mask, self.blank_id)
        loss = (loss / label_mask.sum(1)).mean()
        metrics = {"loss": loss}

        if self.teacher is not None:
            teacher_logits, _ = self.teacher(x, mask, det=True)
            teacher_logits = teacher_logits * (1e10 if self.hard_distill else 1)

            kd_logits = nn.Dense(logits.shape[-1])(hidden)
            loss_kd = optax.softmax_cross_entropy(kd_logits, nn.softmax(teacher_logits))
            loss_kd = (mask * loss_kd).sum() / jnp.maximum(mask.sum(), 1)

            metrics = {
                "loss": loss + 10 * loss_kd,
                "loss_ctc": loss,
                "loss_kd": loss_kd,
            }
        return metrics, {"preds": self.decode_ctc_logits(logits, mask)}


class TrainState(train_state.TrainState):
    ema_decay: float
    ema_params: ArrayTree
    dropout_rng: PRNGKey
    learning_rate_fn: optax.Schedule = flax.struct.field(pytree_node=False)

    def replicate(self) -> TrainState:
        state = flax.jax_utils.replicate(self)
        return state.replace(dropout_rng=shard_prng_key(self.dropout_rng))


@partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
def training_step(state: TrainState, batch: ArrayTree) -> tuple[TrainState, ArrayTree]:
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    rngs = {"dropout": dropout_rng}

    def loss_fn(params: ArrayTree) -> tuple[Array, ArrayTree]:
        metrics, _ = state.apply_fn({"params": params}, **batch, det=False, rngs=rngs)
        return metrics["loss"], metrics

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics, grads = jax.lax.pmean((metrics, grads), "batch")
    state = state.apply_gradients(grads=grads)

    # Update exponential moving average to the model's parameters.
    ema_params = jax.tree_map(
        lambda e, m: e * state.ema_decay + m * (1 - state.ema_decay),
        state.ema_params,
        state.params,
    )
    return state.replace(ema_params=ema_params, dropout_rng=new_dropout_rng), metrics


@partial(jax.pmap, axis_name="batch", static_broadcasted_argnums=(2,))
def validation_step(
    state: TrainState, batch: ArrayTree, use_ema_params: bool = False
) -> tuple[ArrayTree, ArrayTree]:
    params = state.ema_params if use_ema_params else state.params
    metrics, outputs = state.apply_fn({"params": params}, **batch, det=True)
    return jax.lax.pmean(metrics, "batch"), outputs


def load_pretrained_params(params: ArrayTree, checkpoint_path: str) -> ArrayTree:
    with open(checkpoint_path, "rb") as fp:
        pretrained = flax.serialization.msgpack_restore(fp.read())
        if "student" in pretrained:
            pretrained = {"model": pretrained["student"]["encoder"]}
        pretrained = flax.traverse_util.flatten_dict({"model": pretrained["model"]})

    params, num_matches = flax.traverse_util.flatten_dict(params), 0
    for key, value in pretrained.items():
        if key in params and params[key].shape == value.shape:
            params[key], num_matches = value, num_matches + 1

    print(f"[*] {num_matches}/{len(pretrained)} parameters are loaded from checkpoint.")
    return flax.traverse_util.unflatten_dict(params)


def load_teacher_params(
    vocab: dict[str, int], checkpoint_path: str
) -> tuple[Transformer, ArrayTree]:
    with open(checkpoint_path, "rb") as fp:
        params = flax.serialization.msgpack_restore(fp.read())["model"]

    model = Transformer(
        layers=len([x for x in params if x.startswith("layer_")]),
        dim=params["wte"]["kernel"].shape[-1],
        heads=params["layer_0"]["attn"]["wq"]["kernel"].shape[-2],
        labels=len(vocab) + 1,
    )
    return model, params


def create_train_state(args: argparse.Namespace, vocab: dict[str, int]) -> TrainState:
    teacher = None
    if args.teacher_ckpt is not None:
        teacher, teacher_params = load_teacher_params(vocab, args.teacher_ckpt)

    model = Transformer(
        layers=args.layers,
        dim=args.dim,
        heads=args.heads,
        labels=len(vocab) + 1,
        dropout=args.dropout,
        layerdrop=args.layerdrop,
        use_lstm_head=args.use_lstm_head,
    )
    module = TrainModule(
        model=model,
        teacher=teacher,
        blank_id=len(vocab),
        hard_distill=args.hard_distill,
    )

    # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
    # will visualize the summary of the model including parameters.
    init_rngs = {"params": jax.random.PRNGKey(args.init_seed)}
    example_inputs = {
        "landmarks": jnp.zeros((1, args.input_length, args.input_features)),
        "lm_labels": jnp.zeros((1, args.label_length), dtype=jnp.int32),
        "ctc_labels": jnp.zeros((1, args.label_length), dtype=jnp.int32),
    }

    print(module.tabulate(init_rngs, **example_inputs))
    params = module.init(init_rngs, **example_inputs)["params"]

    if args.pretrained_ckpt is not None:
        params = load_pretrained_params(params, args.pretrained_ckpt)
    if teacher is not None:
        params["teacher"] = teacher_params

    # Create learning rate scheduler and optimizer with gradient clipping. The learning
    # rate function will be reused to log current learning rate.
    learning_rate = optax.linear_onecycle_schedule(
        transition_steps=args.training_steps,
        peak_value=args.learning_rate,
        pct_start=args.warmup_steps / args.training_steps,
        pct_final=1.0,
    )
    optimizer = optax.adamw(
        learning_rate,
        b1=args.adam_beta1,
        b2=args.adam_beta2,
        eps=args.adam_eps,
        weight_decay=args.weight_decay,
        mask=partial(jax.tree_util.tree_map, lambda x: x.ndim != 1),
    )
    optimizer = optax.chain(optax.clip_by_global_norm(args.clip_grad_norm), optimizer)
    optimizer = optax.multi_transform(
        {"freeze": optax.set_to_zero(), "trainable": optimizer},
        param_labels=partial(
            jax.tree_util.tree_map_with_path,
            lambda x, _: "freeze" if x[0].key == "teacher" else "trainable",
        ),
    )

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        ema_decay=args.ema_decay,
        ema_params=jax.tree_map(jnp.copy, params),
        tx=optimizer,
        dropout_rng=jax.random.PRNGKey(args.dropout_seed),
        learning_rate_fn=learning_rate,
    )
