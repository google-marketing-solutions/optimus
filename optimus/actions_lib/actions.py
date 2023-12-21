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

from optimus.actions_lib import base_actions


@dataclasses.dataclass(frozen=True)
class _Actions:
  """Instance with and an actions class name, its instance and hyperparameters.

  Attributes:
    name: A name of an actions class.
    instance: An instance of an actions class.
    hyperparameters: Hyperparameters of an actions class.
  """

  name: str
  instance: type[base_actions.BaseActions]
  hyperparameters: config_dict.ConfigDict


_ALL_ACTIONS = config_dict.ConfigDict(
    dict(
        base_actions=_Actions(
            "base_actions",
            base_actions.BaseActions,
            base_actions.DEFAULT_HYPERPARAMETERS,
        )
    )
)


def get_actions(
    *, actions_name: str = "base_actions"
) -> type[base_actions.BaseActions]:
  """Maps the actions name with the corresponding actions class.

  Args:
    actions_name: Actions name.

  Returns:
    The requested actions class.

  Raises:
    LookupError: An error when trying to access an unavailable actions class.
  """
  if actions_name not in _ALL_ACTIONS:
    raise LookupError(f"Unrecognized actions class name: {actions_name}")
  return _ALL_ACTIONS[actions_name].instance


def get_actions_hyperparameters(
    *, actions_name: str = "base_actions"
) -> config_dict.ConfigDict:
  """Maps the actions name with the corresponding actions hyperparameters.

  Args:
    actions_name: Actions name.

  Returns:
    The requested actions class hyperparameters.

  Raises:
    LookupError: An error when trying to access an unavailable actions class.
  """
  if actions_name not in _ALL_ACTIONS:
    raise LookupError(f"Unrecognized actions class name: {actions_name}")
  return _ALL_ACTIONS[actions_name].hyperparameters
