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

"""Registry of all the available actions."""
import dataclasses

from ml_collections.config_dict import config_dict

from optimus.reward_lib import base_reward


@dataclasses.dataclass(frozen=True)
class _Reward:
  """Instance with and a reward class name, its instance and hyperparameters.

  Attributes:
    name: A name of a reward class.
    instance: An instance of a reward class.
    hyperparameters: Hyperparameters of a reward class.
  """

  name: str
  instance: type[base_reward.BaseReward]
  hyperparameters: config_dict.ConfigDict


_ALL_REWARDS = config_dict.ConfigDict(
    dict(
        base_reward=_Reward(
            "base_reward",
            base_reward.BaseReward,
            base_reward.DEFAULT_HYPERPARAMETERS,
        )
    )
)


def get_reward(
    *, reward_name: str = "base_reward"
) -> type[base_reward.BaseReward]:
  """Maps the reward name with the corresponding reward class.

  Args:
    reward_name: Reward name.

  Returns:
    The requested reward class.

  Raises:
    LookupError: An error when trying to access an unavailable reward class.
  """
  if reward_name not in _ALL_REWARDS:
    raise LookupError(f"Unrecognized reward class name: {reward_name}")
  return _ALL_REWARDS[reward_name].instance


def get_reward_hyperparameters(
    *, reward_name: str = "base_reward"
) -> config_dict.ConfigDict:
  """Maps the reward name with the corresponding reward hyperparameters.

  Args:
    reward_name: Reward name.

  Returns:
    The requested reward class hyperparameters.

  Raises:
    LookupError: An error when trying to access an unavailable reward class.
  """
  if reward_name not in _ALL_REWARDS:
    raise LookupError(f"Unrecognized reward class name: {reward_name}")
  return _ALL_REWARDS[reward_name].hyperparameters
