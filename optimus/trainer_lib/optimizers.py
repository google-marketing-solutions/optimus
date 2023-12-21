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

"""Registry of all the available optimizers."""
import dataclasses
from typing import Any, Callable

from ml_collections.config_dict import config_dict
import optax


@dataclasses.dataclass(frozen=True)
class _Optimizer:
  """Instance with an optimizer name and its instance.

  Attributes:
    name: An optimizer name.
    instance: An optimizer instance.
  """

  name: str
  instance: Callable[[Any], optax.GradientTransformation]


_ALL_OPTIMIZERS = config_dict.ConfigDict(
    dict(adam=_Optimizer("adam", optax.adam))
)


def get_optimizer(
    *, optimizer_name: str,
) -> Callable[[Any], optax.GradientTransformation]:
  """Maps the optimizer name with the corresponding optimizer function.

  Args:
    optimizer_name: Optimizer name.

  Returns:
    The requested optimizer function.

  Raises:
    LookupError: An error when trying to access an unavailable optimizer.
  """
  if optimizer_name not in _ALL_OPTIMIZERS:
    raise LookupError(f"Unrecognized optimizer name: {optimizer_name}")
  return _ALL_OPTIMIZERS[optimizer_name].instance
