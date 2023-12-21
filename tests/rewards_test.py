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

"""Tests for utility functions in rewards.py."""
from absl.testing import absltest

from optimus.reward_lib import base_reward
from optimus.reward_lib import rewards


class RewardsTests(absltest.TestCase):

  def test_get_reward(self):
    self.assertEqual(
        rewards.get_reward(reward_name="base_reward"), base_reward.BaseReward
    )

  def test_get_reward_value_error(self):
    with self.assertRaisesRegex(LookupError, "Unrecognized reward class name"):
      rewards.get_reward(reward_name="other_reward")

  def test_get_reward_hyperparameters(self):
    self.assertEqual(
        rewards.get_reward_hyperparameters(reward_name="base_reward"),
        base_reward.DEFAULT_HYPERPARAMETERS,
    )

  def test_get_reward_hyperparameters_value_error(self):
    with self.assertRaisesRegex(LookupError, "Unrecognized reward class name"):
      rewards.get_reward_hyperparameters(reward_name="other_reward")


if __name__ == "__main__":
  absltest.main()
