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

"""Base class for all agents."""

import abc
import dataclasses
from typing import Any, Callable, Final, Mapping, Protocol
import flax
from flax.core import frozen_dict
from flax.training import train_state
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict

_EPSILON: Final[float] = 1e-8
FrozenDict = frozen_dict.FrozenDict


class CalculatePretrainReward(Protocol):

  def __call__(
      self, batch: jnp.ndarray, actions: jnp.ndarray, sign_rewards: bool
  ) -> jnp.ndarray:
    ...


class BaseAgentState(train_state.TrainState):
  """Adds batch_stats to the vanilla Flax TrainState class."""

  batch_stats: flax.core.FrozenDict | dict[str, Any]
  exploration_exploitation_epsilon: float


def apply_policy(
    *,
    parameters: FrozenDict | Mapping[str, Any],
    apply_function: Callable[..., Any],
    batch: jnp.ndarray,
    **kwargs,
) -> tuple[jnp.ndarray, ...]:
  """Applies policy on an input batch.

  Args:
    parameters: The current parameters of the trained agent (model).
    apply_function: An apply function of the trained agent (model).
      batch: An input data batch used to predict an action of shape (batch_size,
        input_dimensions).
    **kwargs: Other arguments, e.g. train and/or mutable.

  Returns:
    The output of an agent's apply_function.
  """
  if "batch_statistics" in kwargs:
    variables = {
        "params": parameters,
        "batch_stats": kwargs["batch_statistics"],
    }
    kwargs.pop("batch_statistics", None)
  else:
    variables = {"params": parameters}
  return apply_function(variables=variables, input_x=batch, **kwargs)


def predict_actions(
    *,
    exploration_exploitation_epsilon: float,
    prng_key: jax.Array,
    probabilities: jnp.ndarray,
) -> jnp.ndarray:
  """Predicts actions depending on the exploration/exploitation rate.

  Args:
    exploration_exploitation_epsilon: An exploration exploitation rate affecting
      if the most optimal actions is selected or not.
    prng_key: A PRNG key to ensure repeatability.
    probabilities: An array with each action probability of shape (,
      total_number_of_possible_actions, ).

  Returns:
    An array with predicted action of shape (1, 1).
  """
  normalized_probabilities = probabilities / jnp.linalg.norm(
      probabilities, 1, axis=0, keepdims=True
  )
  epsilon_adjusted_probabilities = (
      exploration_exploitation_epsilon / len(normalized_probabilities)
      + (1 - exploration_exploitation_epsilon) * normalized_probabilities
  )
  return jax.random.choice(
      key=prng_key, a=probabilities.shape[0], p=epsilon_adjusted_probabilities
  ).reshape(1, 1)


@dataclasses.dataclass(frozen=True)
class PredictionOutput:
  """A mapping of agent experience building blocks and respective arrays.

  Attributes:
    prediction_seed: A prediction seed used to select the action.
    state: An array with state (i.e. features) based on which an agent has to
      predict the most optimal decision. It's of shape (1, input_dimensions)
    action: An array with an action predicted by an agent based on the provided
      context of shape (1, 1).
    value: An array with a "critic" value of shape (1, 1).
    log_probability: An array of log probability of the predicted action of
      shape (1, 1).
    done: An array with booleans indicating if the end of a series of events (if
      applicable, e.g. for TabNet) of shape (1, 1).
    attentive_transformer_loss: An array with attentive transformer losses (if
      applicable, e.g. for TabNet) of shape (1, 1).
  """

  prediction_seed: int
  state: jnp.ndarray
  action: jnp.ndarray
  value: jnp.ndarray
  log_probability: jnp.ndarray
  done: jnp.ndarray
  attentive_transformer_loss: jnp.ndarray


class BaseAgent(metaclass=abc.ABCMeta):
  """Abstract parent class for all agent classes.

  Attributes:
    hyperparameters: The hyperparameteres for the agent class.
    flax_module: A Flax model used for training and making predictions.
  """

  def __init__(self, *, hyperparameters: config_dict.ConfigDict) -> None:
    """Initializes the BaseAgent class.

    Args:
      hyperparameters: Agent's hyperparameters.
    """
    self.hyperparameters = hyperparameters
    self.flax_module = self.build_flax_module()

  @abc.abstractmethod
  def build_flax_module(self) -> Any:
    """Builds a Flax module / model for training."""

  def get_dummy_inputs(self) -> jnp.ndarray:
    """Outputs dummy inputs to initialize a Flax module / model."""
    return jnp.zeros(
        (
            self.hyperparameters.batch_size,
            self.hyperparameters.input_dimensions,
        ),
        dtype=self.hyperparameters.model_data_type,
    )

  def predict(
      self,
      *,
      agent_state: BaseAgentState,
      batch: jnp.ndarray,
      prediction_seed: int,
  ) -> PredictionOutput:
    """Returns a prediction output (incl. actions) given an input batch.

    Args:
      agent_state: Current agent state.
      batch: An input data batch used to predict an action of shape (batch_size,
        input_dimensions).
      prediction_seed: A prediction seed used to select an action.
    """
    (log_probabilities, value, attentive_transformer_loss), _ = apply_policy(
        parameters=agent_state.params,
        batch_statistics=agent_state.batch_stats,
        apply_function=agent_state.apply_fn,
        batch=batch,
        train=True,
        mutable=["batch_stats"],
    )
    predicted_action = predict_actions(
        exploration_exploitation_epsilon=agent_state.exploration_exploitation_epsilon,
        prng_key=jax.random.PRNGKey(prediction_seed),
        probabilities=jnp.exp(log_probabilities).flatten(),
    )
    log_probabilities = jnp.take(log_probabilities, predicted_action)
    attentive_transformer_loss = jnp.asarray(
        attentive_transformer_loss
    ).reshape(1, 1)
    if self.hyperparameters.replace_nans_in_prediction:
      value = jnp.nan_to_num(value, nan=_EPSILON)
      log_probabilities = jnp.nan_to_num(log_probabilities, nan=_EPSILON)
      attentive_transformer_loss = jnp.nan_to_num(
          attentive_transformer_loss, nan=_EPSILON
      )
    return PredictionOutput(
        prediction_seed=prediction_seed,
        state=batch,
        action=predicted_action,
        value=value,
        log_probability=log_probabilities,
        done=jnp.repeat(True, 1).reshape(1, 1),
        attentive_transformer_loss=attentive_transformer_loss,
    )

  def evaluate_predict(
      self,
      *,
      agent_state: BaseAgentState,
      batch: jnp.ndarray,
  ) -> jnp.ndarray:
    """Returns predicted actions given an input batch.

    Args:
      agent_state: Current agent state.
      batch: An input data batch used to predict an action of shape (batch_size,
        input_dimensions).
    """
    log_probabilities, _, _ = apply_policy(
        parameters=agent_state.params,
        apply_function=agent_state.apply_fn,
        batch=batch,
        batch_statistics=agent_state.batch_stats,
        train=False,
        mutable=False,
    )
    return jnp.argmax(log_probabilities, axis=1).reshape(batch.shape[0], 1)

  def pretrain_predict(
      self,
      *,
      batch: jnp.ndarray,
      agent_state: BaseAgentState,
      calculate_pretrain_reward: CalculatePretrainReward,
      prediction_seed: int,
  ) -> jnp.ndarray:
    """Returns a prediction for a single observation during pre-training.

    Args:
      batch: An array with with a unique user id and user data.
      agent_state: The current agent state.
      calculate_pretrain_reward: A function that calculates (or predicts)
        rewards for actions taken during pre-training process.
      prediction_seed: A prediction seed used to select the action.

    Raises:
      ValueError: An error when the input batch isn't one-dimensional.
    """
    if batch.ndim != 1:
      raise ValueError(
          "The input batch must be one-dimensional. There are"
          f" {batch.ndim} dimensions now."
      )
    context = jnp.expand_dims(
        batch[: self.hyperparameters.input_dimensions], axis=0
    )
    if isinstance(prediction_seed, jnp.ndarray):
      prediction_seed = prediction_seed.reshape().astype(jnp.int32)
    pretraining_prediction_output = self.predict(
        agent_state=agent_state,
        batch=context,
        prediction_seed=prediction_seed,
    )
    reward = calculate_pretrain_reward(
        batch=batch,
        actions=pretraining_prediction_output.action,
        sign_rewards=self.hyperparameters.sign_rewards,
    )
    return jnp.concatenate(
        [
            pretraining_prediction_output.state,
            pretraining_prediction_output.action,
            reward,
            pretraining_prediction_output.value,
            pretraining_prediction_output.log_probability,
            pretraining_prediction_output.done,
            pretraining_prediction_output.attentive_transformer_loss,
        ],
        axis=1,
    ).flatten()
