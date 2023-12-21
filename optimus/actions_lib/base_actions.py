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

"""Base class for all actions."""
import abc

from ml_collections.config_dict import config_dict

DEFAULT_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        action_space=config_dict.placeholder(list),
        actions_name="base_actions",
    )
)


class BaseActions(metaclass=abc.ABCMeta):
  """Abstract parent class for all actions classes.

  Attributes:
    hyperparameters: The hyperparameteres for the action class.
  """

  def __init__(self, *, hyperparameters: config_dict.ConfigDict) -> None:
    """Initializes the BaseActions class.

    Args:
      hyperparameters: BaseActions'/ experiment hyperparameters.
    """
    self.hyperparameters = hyperparameters
