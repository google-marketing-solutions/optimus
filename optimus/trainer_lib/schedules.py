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

"""Registry of all the available schedules."""
import dataclasses
from typing import Any, Callable

from ml_collections.config_dict import config_dict
import optax


def check_schedule_hyperparameters(
    *,
    schedule_hyperparameters: config_dict.ConfigDict,
    expected_keys: tuple[str, ...],
    other_keys: tuple[str, ...] = ("schedule",),
) -> None:
  """Checks if all required schedule hyperparameters are present in the experiment dictionary.

  Args:
    schedule_hyperparameters: The schedule hyperparameters.
    expected_keys: A list with required schedule hyperparameters.
    other_keys: A tuple of keys that are part of schedule hyperparameters,
      but not required by the schedule function.

  Raises:
    ValueError: An error when schedule hyperparameters are invalid.
  """
  expected_keys = expected_keys + other_keys
  if set(schedule_hyperparameters.keys()) != set(expected_keys):
    raise ValueError(
        "Provided schedule_hyperparameters keys are invalid. Received:"
        f" {sorted(schedule_hyperparameters)}, Expected:"
        f" {sorted(expected_keys)}"
    )


def linear_schedule(
    schedule_hyperparameters: config_dict.ConfigDict,
) -> Callable[[Any], Any]:
  """Returns a linear schedule function.

  Args:
    schedule_hyperparameters: The schedule hyperparameters.
  """
  expected_keys = tuple(["initial_value", "end_value", "transition_steps"])
  check_schedule_hyperparameters(
      schedule_hyperparameters=schedule_hyperparameters,
      expected_keys=expected_keys,
  )
  linear_schedule_hyperparameters = {
      "init_value": schedule_hyperparameters.initial_value,
      "end_value": schedule_hyperparameters.end_value,
      "transition_steps": schedule_hyperparameters.transition_steps,
  }
  return optax.linear_schedule(**linear_schedule_hyperparameters)


def constant_schedule(
    schedule_hyperparameters: config_dict.ConfigDict,
) -> Callable[[Any], Any]:
  """Returns a constant schedule function.

  Args:
    schedule_hyperparameters: The schedule hyperparameters.
  """
  expected_keys = tuple(["initial_value"])
  check_schedule_hyperparameters(
      schedule_hyperparameters=schedule_hyperparameters,
      expected_keys=expected_keys,
  )
  return optax.constant_schedule(
      value=schedule_hyperparameters.initial_value
  )


def warmup_exponential_decay_schedule(
    schedule_hyperparameters: config_dict.ConfigDict,
) -> Callable[[Any], Any]:
  """Returns a warmup exponential decay schedule schedule function.

  Args:
    schedule_hyperparameters: The schedule hyperparameters.
  """
  expected_keys = tuple([
      "initial_value",
      "peak_value",
      "warmup_steps",
      "transition_steps",
      "decay_rate",
      "transition_begin",
      "staircase",
      "end_value",
  ])
  check_schedule_hyperparameters(
      schedule_hyperparameters=schedule_hyperparameters,
      expected_keys=expected_keys,
  )
  warmup_exponential_decay_schedule_hyperparameters = {
      "init_value": schedule_hyperparameters.initial_value,
      "peak_value": schedule_hyperparameters.peak_value,
      "warmup_steps": schedule_hyperparameters.warmup_steps,
      "transition_steps": schedule_hyperparameters.transition_steps,
      "decay_rate": schedule_hyperparameters.decay_rate,
      "transition_begin": schedule_hyperparameters.transition_begin,
      "staircase": schedule_hyperparameters.staircase,
      "end_value": schedule_hyperparameters.end_value,
  }
  return optax.warmup_exponential_decay_schedule(
      **warmup_exponential_decay_schedule_hyperparameters
  )


@dataclasses.dataclass(frozen=True)
class _Schedule:
  """Instance with a schedule name and its instance.

  Attributes:
    name: A schedule name.
    instance: A schedule instance.
  """

  name: str
  instance: Any


_ALL_SCHEDULES = config_dict.FrozenConfigDict(
    dict(
        linear_schedule=_Schedule("linear_schedule", linear_schedule),
        constant_schedule=_Schedule("constant_schedule", constant_schedule),
        warmup_exponential_decay_schedule=_Schedule(
            "warmup_exponential_decay_schedule",
            warmup_exponential_decay_schedule,
        ),
    ),
)


def get_schedule(
    *,
    schedule_name: str,
) -> Callable[[config_dict.ConfigDict], Callable[[float], float]]:
  """Maps the schedule name with the corresponding schedule function.

  Args:
    schedule_name: Schedule name.

  Returns:
    The requested schedule function.

  Raises:
    LookupError: An error when trying to access an unavailable schedule.
  """
  if schedule_name not in _ALL_SCHEDULES:
    raise LookupError(f"Unrecognized schedule name: {schedule_name}")
  return _ALL_SCHEDULES[schedule_name].instance
