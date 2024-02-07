# Copyright 2024 Google LLC.
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

"""Template for a custom reward class."""

import jax.numpy as jnp
from optimus.reward_lib import base_reward
import tensorflow as tf


class CustomReward(base_reward.BaseReward):
  """A custom reward class managing the reward calculation proceess.

  Attributes:
    hyperparameters: A rewards class hyperparameters.
  """

  def calculate_reward(
      self, actions: tf.Tensor, reactions: tf.Tensor, sign_rewards: bool
  ) -> tf.Tensor:
    """Returns a reward given the predicted actions and end-user reactions.

    This function must be written in TensorFlow. A reaction can compromise of
    multiple numerical
    data points translated through an custom logic to a final numerical award.

    Args:
      actions: An array with predicted most optimal actions. It should be of
        shape (batch_size, 1).
      reactions: An array with all the data points collected as feedback from
        an end-user. In this case, it's just a representation of the known
        most optimal action.
      sign_rewards: An indicator if to sign the final reward or not.
    """
    reward = tf.where(actions == reactions, 1.0, 0.0)
    if sign_rewards:
      reward = tf.math.sign(reward)
    return reward

  def calculate_pretrain_reward(
      self, batch: jnp.ndarray, actions: jnp.ndarray, sign_rewards: bool
  ) -> jnp.ndarray:
    """Returns a reward given the predicted actions during pre training.

    The Optimus model can be pretrained assuming there is a pre-exisitng dataset
    with examples containing features (that later will be used in production)
    and corresponding most optimal actions. Pretraining make Optimus able ot
    make less random decisions early after its deployment.

    Args:
      batch: An array with features. It should be of shape (batch_size,
        input_dimensions).
      actions: An array with known most optimal actions. It should be of
        shape (batch_size, 1).
      sign_rewards: An indicator if to sign the final reward or not.
    """
    reactions = batch[-1:]
    rewards = jnp.where(reactions == actions, 1.0, 0.0)
    if sign_rewards:
      rewards = jnp.sign(rewards)
    return rewards
