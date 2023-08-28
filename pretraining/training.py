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
from typing import Callable

import flax
import flax.linen as nn
import flax.struct
import jax
import jax.numpy as jnp
import optax
from chex import Array, ArrayTree, PRNGKey
from flax.training import train_state
from flax.training.common_utils import shard_prng_key

from modeling import ConvStack, Data2vecStudent, Data2vecTeacher, Transformer


class TrainModule(nn.Module):
    teacher: Data2vecTeacher
    student: Data2vecStudent

    def compute_masked_stdev(self, x: Array, mask: Array) -> Array:
        x, mask = x.reshape(-1, x.shape[-1]), mask.reshape(-1, 1)
        lengths = jnp.maximum(mask.sum(), 1.0)

        x = x - (x * mask).sum(0) / lengths
        return jnp.sqrt((x**2 * mask).sum(0) / lengths + 1e-10).mean()

    def __call__(self, landmarks: Array) -> ArrayTree:
        x_student, mask = self.student(landmarks)
        x_teacher = self.teacher(landmarks)

        x_student = x_student.reshape(x_teacher.shape[0], -1, *x_student.shape[1:])
        mask = mask.reshape(x_student.shape[:-1])

        loss = jnp.square(x_teacher[:, None] - x_student).mean(-1)
        loss = (loss * mask).sum() / mask.sum()

        return {
            "loss": loss,
            "student_stdev": self.compute_masked_stdev(x_student, mask),
            "teacher_stdev": self.compute_masked_stdev(x_teacher, mask.any(1)),
        }


class TrainState(train_state.TrainState):
    mask_rng: PRNGKey
    dropout_rng: PRNGKey
    ema_decay_fn: Callable[[int], float] = flax.struct.field(pytree_node=False)
    learning_rate_fn: optax.Schedule = flax.struct.field(pytree_node=False)

    def replicate(self) -> TrainState:
        state = flax.jax_utils.replicate(self)
        state = state.replace(mask_rng=shard_prng_key(self.mask_rng))
        state = state.replace(dropout_rng=shard_prng_key(self.dropout_rng))
        return state


@partial(jax.pmap, axis_name="batch", donate_argnums=(0,))
def training_step(state: TrainState, batch: Array) -> tuple[TrainState, ArrayTree]:
    mask_rng, new_mask_rng = jax.random.split(state.mask_rng)
    dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)
    rngs = {"mask": mask_rng, "dropout": dropout_rng}

    def loss_fn(params: ArrayTree) -> tuple[Array, ArrayTree]:
        metrics = state.apply_fn({"params": params}, batch, rngs=rngs)
        return metrics["loss"], metrics

    (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
    metrics, grads = jax.lax.pmean((metrics, grads), "batch")
    state = state.apply_gradients(grads=grads)

    # Update the teacher model by applying exponential moving average to the student's
    # parameters. The averaging weight will be changed by current training step.
    params = jax.tree_map(
        lambda t, s: t * (gamma := state.ema_decay_fn(state.step)) + s * (1 - gamma),
        state.params["teacher"]["encoder"],
        state.params["student"]["encoder"],
    )
    state = state.replace(
        params={"teacher": {"encoder": params}, "student": state.params["student"]},
        mask_rng=new_mask_rng,
        dropout_rng=new_dropout_rng,
    )
    return state, metrics


def create_train_state(args: argparse.Namespace) -> TrainState:
    encoder_kwargs = dict(
        layers=args.encoder_layers,
        dim=args.encoder_dim,
        heads=args.encoder_heads,
        dropout=args.encoder_dropout,
        layerdrop=args.encoder_layerdrop,
    )
    decoder_kwargs = dict(
        layers=args.decoder_layers,
        dim=args.decoder_dim,
        kernel=args.decoder_kernel,
        labels=args.encoder_dim,
        dropout=args.decoder_dropout,
        layerdrop=args.decoder_layerdrop,
    )

    teacher = Data2vecTeacher(Transformer(**encoder_kwargs), args.average_layers)
    student = Data2vecStudent(
        Transformer(**encoder_kwargs),
        ConvStack(**decoder_kwargs),
        num_masks=args.num_masks,
        mask_block=args.mask_block,
        mask_prob=args.mask_prob,
        mask_adjust=args.mask_adjust,
        mask_stdev=args.mask_stdev,
    )
    module = TrainModule(teacher, student)

    # Initialize the model weights with dummy inputs. Using the init RNGS and inputs, we
    # will visualize the summary of the model including parameters.
    init_rngs = {
        "params": jax.random.PRNGKey(args.init_seed),
        "mask": jax.random.PRNGKey(args.mask_seed),
        "dropout": jax.random.PRNGKey(args.dropout_seed),
    }
    example_inputs = jnp.zeros((1, args.input_length, args.input_features))

    print(module.tabulate(init_rngs, example_inputs))
    params = module.init(init_rngs, example_inputs)["params"]
    params["teacher"]["encoder"] = jax.tree_map(jnp.copy, params["student"]["encoder"])

    # Create learning rate scheduler and optimizer with gradient clipping. While the
    # teacher model is exponentially moving-averaged from student model, gradients for
    # teacher model will be set to zero.
    learning_rate = optax.cosine_onecycle_schedule(
        transition_steps=args.training_steps,
        peak_value=args.learning_rate,
        pct_start=args.warmup_steps / args.training_steps,
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
        {"teacher": optax.set_to_zero(), "student": optimizer},
        param_labels=partial(jax.tree_util.tree_map_with_path, lambda x, _: x[0].key),
    )

    def ema_decay_fn(step: Array) -> Array:
        interp = jnp.minimum(step / args.ema_steps, 1.0)
        return args.ema_end * interp + args.ema_start * (1 - interp)

    return TrainState.create(
        apply_fn=module.apply,
        params=params,
        tx=optimizer,
        mask_rng=jax.random.PRNGKey(args.mask_seed),
        dropout_rng=jax.random.PRNGKey(args.dropout_seed),
        ema_decay_fn=ema_decay_fn,
        learning_rate_fn=learning_rate,
    )
