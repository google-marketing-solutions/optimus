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

"""Tests for utility functions in base_reward.py."""
from absl.testing import absltest
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf

from optimus.reward_lib import base_reward

_TEST_BASE_REWARD_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        sign_rewards=True,
    )
)


class MockBaseReward(base_reward.BaseReward):

  def __init__(self, hyperparameters=_TEST_BASE_REWARD_HYPERPARAMETERS):
    super().__init__(hyperparameters=hyperparameters)

  def calculate_reward(self, predicted_actions, reactions, sign_rewards):
    del predicted_actions, reactions, sign_rewards
    return tf.constant(0)

  def calculate_pretrain_reward(self, batch, actions, sign_rewards):
    del batch, actions
    return np.array(0)

  def calculate_evaluation_reward(self, predicted_actions, target_actions):
    return np.array(0)


class BaseRewardTests(absltest.TestCase):

  def test_base_reward(self):
    self.assertIsInstance(
        MockBaseReward(),
        base_reward.BaseReward,
    )

  def test_calculate_evaluation_reward(self):
    actual_output = MockBaseReward().calculate_evaluation_reward(
        predicted_actions=jnp.asarray([[1]]), target_actions=jnp.asarray([[1]])
    )
    self.assertEqual(actual_output.item(), 0)


if __name__ == "__main__":
  absltest.main()
