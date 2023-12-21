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

"""Base class for all rewards."""

import abc

import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import tensorflow as tf

DEFAULT_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        sign_rewards=True,
        reward_name="base_reward",
    )
)


class BaseReward(metaclass=abc.ABCMeta):
  """Abstract parent class for all rewards classes.

  Attributes:
    hyperparameters: A rewards class hyperparameters.
  """

  def __init__(self, *, hyperparameters: config_dict.ConfigDict) -> None:
    """Initalizes the BaseReward class.

    Args:
      hyperparameters: The hyperparameteres for the reward class.
    """
    self.hyperparameters = hyperparameters

  @abc.abstractmethod
  def calculate_reward(
      self, actions: tf.Tensor, reactions: tf.Tensor, sign_rewards: bool
  ) -> tf.Tensor:
    """Calculates rewards given the selected actions and end-user reactions."""

  @abc.abstractmethod
  def calculate_pretrain_reward(
      self, batch: jnp.ndarray, actions: jnp.ndarray, sign_rewards: bool
  ) -> jnp.ndarray:
    """Calculates pretrain rewards given the selected actions."""

  def calculate_evaluation_reward(
      self,
      *,
      predicted_actions: jnp.ndarray,
      target_actions: jnp.ndarray,
  ) -> jnp.ndarray:
    """Calculates evaluation rewards given the selected and target actions.

    Args:
      predicted_actions: A batch with selected actions to be evaluated of shape
        (batch_size, 1).
      target_actions: An array with target actions of shape (batch_size, 1).

    Returns:
      A batch with calculated evaluation rewards.
    """
    return jnp.where(predicted_actions == target_actions, 1.0, -1.0)
