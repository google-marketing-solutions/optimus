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

"""Registry of all the available agents."""
import dataclasses
from typing import TypeVar

from ml_collections.config_dict import config_dict

from optimus.agent_lib import base_agent
from optimus.agent_lib import tabnet

_BaseAgentType = TypeVar("_BaseAgentType", bound=base_agent.BaseAgent)


@dataclasses.dataclass(frozen=True)
class _Agent:
  """Instance with and an agent class name, its instance and hyperparameters.

  Attributes:
    name: A name of an agent class.
    instance: An instance of an agent class.
    hyperparameters: Hyperparameters of an agent class.
  """

  name: str
  instance: object
  hyperparameters: config_dict.ConfigDict


_ALL_AGENTS = config_dict.ConfigDict(
    dict(
        tabnet=_Agent(
            "tabnet", tabnet.TabNetAgent, tabnet.DEFAULT_HYPERPARAMETERS
        )
    )
)


def get_agent(*, agent_name: str = "tabnet") -> _BaseAgentType:
  """Maps the agent name with the corresponding agent class.

  Args:
    agent_name: Agent name.

  Returns:
    The requested agent class.

  Raises:
    LookupError: An error when trying to access an unavailable agent class.
  """
  if agent_name not in _ALL_AGENTS:
    raise LookupError(f"Unrecognized agent class name: {agent_name}")
  return _ALL_AGENTS[agent_name].instance


def get_agent_hyperparameters(
    *, agent_name: str = "tabnet"
) -> config_dict.ConfigDict:
  """Maps the agent name with the corresponding agent hyperparameters.

  Args:
    agent_name: Agent name.

  Returns:
    The requested agent class hyperparameters.

  Raises:
    LookupError: An error when trying to access an unavailable agent class.
  """
  if agent_name not in _ALL_AGENTS:
    raise LookupError(f"Unrecognized agent class name: {agent_name}")
  return _ALL_AGENTS[agent_name].hyperparameters
