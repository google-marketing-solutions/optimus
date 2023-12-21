# Copyright 2023 Google LLC.
#
# Licensed under the MIT License;
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.mit.edu/~amini/LICENSE.md
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Registry of all the available losses."""
import dataclasses
from typing import Any, Callable, Mapping

import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

from optimus.agent_lib import base_agent


@dataclasses.dataclass()
class Trajectories:
  """Trajectories from an input batch.

  Attributes:
    context_state: Context state array of shape (batch_size, input_dimensions).
    actions: Actions array of shape (batch_size, action_space_length).
    original_log_probabilities: Original log probabilities array of shape
      (batch_size, action_space_length).
    returns: Returns array of shape (batch_size, ).
    advantages: Advantages array of shape (batch_size, ).
    attentive_transformer_losses: Attentive transformer losses array of shape
      (batch_size, ).
  """

  context_state: jnp.ndarray
  actions: jnp.ndarray
  original_log_probabilities: jnp.ndarray
  returns: jnp.ndarray
  advantages: jnp.ndarray
  attentive_transformer_losses: jnp.ndarray


def process_batch(
    *,
    batch: jnp.ndarray,
    hyperparameters: config_dict.ConfigDict,
) -> Trajectories:
  """Process a batch with trajectories.

  Args:
    batch: An input batch.
    hyperparameters: Experiment hyperparameters.

  Returns:
    Trajectories dataclass with the processed input data array.
  """
  context_state = batch[:, 0 : hyperparameters.input_dimensions]
  if hyperparameters.action_space_length > 1:
    skip_indexes_length = (
        hyperparameters.input_dimensions + hyperparameters.action_space_length
    )
    actions = jnp.array(
        batch[:, hyperparameters.input_dimensions : skip_indexes_length],
        dtype=jnp.int32,
    )
    actions = jnp.array_split(
        actions, hyperparameters.action_space_length, axis=1
    )
  else:
    skip_indexes_length = hyperparameters.input_dimensions
    actions = jnp.array(
        batch[:, hyperparameters.input_dimensions], dtype=jnp.int32
    )
  original_log_probabilities = batch[:, skip_indexes_length + 1]
  returns = batch[:, skip_indexes_length + 2]
  advantages = batch[:, skip_indexes_length + 3]
  attentive_transformer_losses = batch[:, skip_indexes_length + 4]
  return Trajectories(
      context_state,
      actions,
      original_log_probabilities,
      returns,
      advantages,
      attentive_transformer_losses,
  )


def calculate_entropy_and_ratios(
    predicted_log_probabilities: jnp.ndarray,
    *,
    actions: jnp.ndarray,
    original_log_probabilities: jnp.ndarray,
    action_space_length: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Calculate entropy and ratios.

  Computes a loss as a sum of 3 components: the negative of the PPO clipped
  surrogate objective, the value function loss and the negative of the entropy
  bonus. For more on PPO see:

  https://arxiv.org/abs/1707.06347

  The loss is then modified with lambda_sparse and m_losses, required for
  TabNet training.

  Args:
    predicted_log_probabilities: An array with log probabilities from the
      apply_policy method in the loss calculations process.
    actions: An array with selected actions.
    original_log_probabilities: An array with log probabilities of selected
      actions.
    action_space_length: The number of actions sets.

  Returns:
    A tuple where the first element is the entropy between action
    probabilities and log probabilities and the second element is
    the ratios between log probabilities of taken actions and original log
    probabilities array of shape (batch_size, ).
  """
  if action_space_length > 1:
    predicted_probabilities = jax.tree_util.tree_map(
        jnp.exp, predicted_log_probabilities
    )
    entropy = jax.tree_util.tree_map(
        lambda x, y: jnp.sum(-x * y, axis=1).mean(),
        predicted_probabilities,
        predicted_log_probabilities,
    )
    entropy = jnp.mean(jnp.array(entropy, dtype=jnp.float32), axis=0)
    log_probs_action_taken = [
        jax.vmap(lambda p, a: p[a])(sub_probs, sub_actions).flatten()
        for sub_probs, sub_actions in zip(predicted_log_probabilities, actions)
    ]
    ratios = jax.tree_util.tree_map(
        lambda x, y: jnp.exp(x - y),
        log_probs_action_taken,
        list(jnp.repeat(original_log_probabilities, action_space_length)),
    )
    ratios = jnp.mean(jnp.array(ratios, dtype=jnp.float32), axis=0)
  else:
    predicted_probabilities = jnp.exp(predicted_log_probabilities)
    entropy = jnp.array(
        jnp.sum(
            -predicted_probabilities * predicted_log_probabilities, axis=1
        ).mean(),
        dtype=jnp.float32,
    )
    log_probabilities_action_taken = jax.vmap(lambda lp, a: lp[a])(
        predicted_log_probabilities, actions
    )
    ratios = jnp.exp(
        log_probabilities_action_taken - original_log_probabilities
    )
  return (entropy, ratios)


def calculate_tabnet_proximal_policy_optimization_loss(
    parameters: base_agent.FrozenDict | Mapping[str, Any],
    batch_statistics: base_agent.FrozenDict | Mapping[str, Any],
    apply_function: Callable[..., Any],
    *,
    batch: jnp.ndarray,
    clip_parameters_coefficient: float,
    hyperparameters: config_dict.ConfigDict,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Calculate loss and update gradients.

  Computes a loss as a sum of 3 components: the negative of the PPO clipped
  surrogate objective, the value function loss and the negative of the entropy
  bonus. For more on PPO see:

  https://arxiv.org/abs/1707.06347

  The loss is then modified with lambda sparse and attentive transformer loss,
  required for TabNet training.

  Args:
    parameters: The current parameters of the trained agent (model).
    batch_statistics: The current batch statistics of the trained agent (model).
    apply_function: An apply function of the trained agent (model).
    batch: An input batch.
    clip_parameters_coefficient: The clipping coefficient used to clamp ratios
      in loss function.
    hyperparameters: Experiment hyperparameters.

  Returns:
    A tuple where the first element is step loss and the second
    element is the updated batch stats.
  """
  processed_batch = process_batch(batch=batch, hyperparameters=hyperparameters)
  (
      predicted_log_probabilities,
      values,
      attentive_transformer_losses,
  ), updated_batch_stats = base_agent.apply_policy(
      parameters=parameters,
      apply_function=apply_function,
      batch=processed_batch.context_state,
      batch_statistics=batch_statistics,
      train=True,
      mutable=["batch_stats"],
  )
  values = values[:, 0]
  value_loss = jnp.mean(jnp.square(processed_batch.returns - values), axis=0)
  entropy, ratios = calculate_entropy_and_ratios(
      predicted_log_probabilities,
      actions=processed_batch.actions,
      original_log_probabilities=processed_batch.original_log_probabilities,
      action_space_length=hyperparameters.action_space_length,
  )
  advantages = (
      processed_batch.advantages - processed_batch.advantages.mean()
  ) / (processed_batch.advantages.std() + 1e-8)
  pg_loss = ratios * advantages
  clipped_loss = advantages * jax.lax.clamp(
      1.0 - clip_parameters_coefficient,
      ratios,
      1.0 + clip_parameters_coefficient,
  )
  proximal_policy_optimization_loss = -jnp.mean(
      jnp.minimum(pg_loss, clipped_loss), axis=0
  )
  final_loss = (
      proximal_policy_optimization_loss
      + hyperparameters.value_function_coefficient * value_loss
      - hyperparameters.entropy_coefficient * entropy
  )
  attentive_transformer_losses = jnp.mean(attentive_transformer_losses)
  final_loss = (
      final_loss - hyperparameters.lambda_sparse * attentive_transformer_losses
  )
  return final_loss, updated_batch_stats


@dataclasses.dataclass(frozen=True)
class _Loss:
  """Instance with a loss name and its instance.

  Attributes:
    name: A loss name.
    instance: A loss instance.
  """

  name: str
  instance: Any


_ALL_LOSSES = config_dict.FrozenConfigDict(
    dict(
        tabnet_proximal_policy_optimization_loss=_Loss(
            "tabnet_proximal_policy_optimization_loss",
            calculate_tabnet_proximal_policy_optimization_loss,
        )
    )
)


def get_loss(
    *,
    loss_name: str,
) -> Callable[[config_dict.ConfigDict], Callable[[float], float]]:
  """Maps the loss name with the corresponding loss function.

  Args:
    loss_name: A loss name.

  Returns:
    The requested loss function.

  Raises:
    LookupError: An error when trying to access an unavailable loss.
  """
  if loss_name not in _ALL_LOSSES:
    raise LookupError(f"Unrecognized loss name: {loss_name}")
  return _ALL_LOSSES[loss_name].instance
