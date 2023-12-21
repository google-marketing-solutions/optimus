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

"""Registry of all the available trainers."""
import dataclasses

from ml_collections.config_dict import config_dict

from optimus.trainer_lib import base_trainer


@dataclasses.dataclass(frozen=True)
class _Trainer:
  """Instance with artifact name, the corresponding class and hyperparameters.

  Attributes:
    name: A trainer class name.
    instance: A trainer class instance.
    hyperparameters: A trainer class hyperparameters.
  """

  name: str
  instance: type[base_trainer.BaseTrainer]
  hyperparameters: config_dict.ConfigDict


_ALL_TRAINERS = config_dict.ConfigDict(
    dict(
        base_trainer=_Trainer(
            "base_trainer",
            base_trainer.BaseTrainer,
            base_trainer.DEFAULT_HYPERPARAMETERS,
        )
    )
)


def get_trainer(
    *, trainer_name: str = "base_trainer"
) -> type[base_trainer.BaseTrainer]:
  """Maps the trainer name with the corresponding trainer class.

  Args:
    trainer_name: Trainer name.

  Returns:
    The requested trainer class.

  Raises:
    LookupError: An error when trying to access an unavailable trainer class.
  """
  if trainer_name not in _ALL_TRAINERS:
    raise LookupError(f"Unrecognized trainer class name: {trainer_name}")
  return _ALL_TRAINERS[trainer_name].instance


def get_trainer_hyperparameters(
    *, trainer_name: str = "base_trainer"
) -> config_dict.ConfigDict:
  """Maps the trainer name with the corresponding trainer hyperparameters.

  Args:
    trainer_name: Trainer name.

  Returns:
    The requested trainer class hyperparameters.

  Raises:
    LookupError: An error when trying to access an unavailable trainer class.
  """
  if trainer_name not in _ALL_TRAINERS:
    raise LookupError(f"Unrecognized trainer class name: {trainer_name}")
  return _ALL_TRAINERS[trainer_name].hyperparameters
