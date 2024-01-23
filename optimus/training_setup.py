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

"""Utility functions for the train operation."""
import itertools
import json
import os
from typing import Any, Final, Mapping, Sequence

from absl import logging
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import portpicker
import tensorflow as tf

_CPU: Final[str] = "cpu"
_GPU: Final[str] = "gpu"
_TPU: Final[str] = "tpu"
_VALID_ACCELERATORS = (_CPU, _GPU, _TPU)
_NESTED_HYPERPARAMETERS = (
    "optimizer_hyperparameters",
    "learning_rate_hyperparameters",
    "exploration_exploitation_hyperparameters",
)


def _verify_action_hyperparameters(
    *,
    hyperparameters: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
  """Verifies if all the required action hyperparameters are present.

  Args:
    hyperparameters: Experiment hyperparameters.

  Returns:
    Verified experiment hyperparameters.
  """
  action_space = hyperparameters.action_space
  single_task_training = len(action_space) == 1
  with hyperparameters.unlocked():
    del hyperparameters["action_space"]
    hyperparameters["action_space"] = (
        action_space[0] if single_task_training else tuple(action_space)
    )
    hyperparameters["action_space_length"] = (
        1 if single_task_training else len(action_space)
    )
  return hyperparameters


def _verify_trainer_hyperparameters(
    *,
    hyperparameters: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
  """Verifies if all the required trainer hyperparameters are present."""
  if "train_steps" not in hyperparameters:
    raise ValueError("train_steps hyperparameter missing.")
  return hyperparameters


def _aggregate_core_hyperparameters(
    *,
    actions_hyperparameters: config_dict.ConfigDict,
    agent_hyperparameters: config_dict.ConfigDict,
    reward_hyperparameters: config_dict.ConfigDict,
    data_pipeline_hyperparameters: config_dict.ConfigDict,
    trainer_hyperparameters: config_dict.ConfigDict,
) -> tuple[dict[str, Any], ...]:
  """Aggregates core hyperparameters.

  Args:
    actions_hyperparameters: A mapping between action class hyperparameters and
      values.
    agent_hyperparameters: A mapping between agent class hyperparameters and
      values.
    reward_hyperparameters: A mapping between reward class hyperparameters and
      values.
    data_pipeline_hyperparameters: A mapping between data pipeline class
      hyperparameters and values.
    trainer_hyperparameters: A mapping between trainer class hyperparameters and
      values.

  Returns:
    Aggregated core hyperparameters.
  """
  hyperparameters_sets = (
      actions_hyperparameters,
      agent_hyperparameters,
      reward_hyperparameters,
      data_pipeline_hyperparameters,
      trainer_hyperparameters,
  )
  return tuple(
      hyperparameter_set.to_dict()
      for hyperparameter_set in hyperparameters_sets
  )


def _merge_hyperparameters(
    *,
    hyperparameters: Sequence[Mapping[str, Any]],
) -> config_dict.ConfigDict:
  """Returns a merged version of hyperparameter mappings.

  Args:
    hyperparameters: A sequence of hyperparameter mappings to be merged.

  Raises:
    ValueError: An error when there's an overlap between merged hyperparameter
      dictionaries.
  """
  merged_hyperparameters = {}
  merged_hyperparameters_keys = []
  for hyperparameter_subset in hyperparameters:
    hyperparameter_subset_keys = list(hyperparameter_subset.keys())
    overlapping_hyperparameters = tuple([
        i
        for i in hyperparameter_subset_keys
        if i in merged_hyperparameters_keys
    ])
    if overlapping_hyperparameters:
      raise ValueError(
          "There is an overlap in the provided hyperparameters. The repeated"
          f" hyperparameters are: {overlapping_hyperparameters}"
      )
    merged_hyperparameters.update(hyperparameter_subset)
    merged_hyperparameters_keys += hyperparameter_subset_keys
  return config_dict.ConfigDict(merged_hyperparameters).lock()


def _convert_nested_hyperparameters(
    *,
    hyperparameters: config_dict.ConfigDict,
) -> config_dict.ConfigDict:
  """Converts all nested hyperparameters to the desired format.

  Args:
    hyperparameters: All the experiment hyperparameters.

  Returns:
    Experiment hyperparameters with converted the optimizer and learning rate
    hyperparameters values.
  """
  for key in _NESTED_HYPERPARAMETERS:
    if key not in hyperparameters:
      with hyperparameters.unlocked():
        hyperparameters[key] = config_dict.ConfigDict()
  for key in _NESTED_HYPERPARAMETERS:
    hyperparameters[key].unlock()
  return hyperparameters


def _update_hyperparameters_from_overrides(
    *,
    hyperparameters: config_dict.ConfigDict,
    manual_overrides: str | None = None,
    file_overrides: str | None = None,
) -> config_dict.ConfigDict:
  """Updates the experiment hyperparameters with hyperparameter overrides.

  Args:
    hyperparameters: The unmodified experiment hyperparameters.
    manual_overrides: A JSON string with hyperparameters overrides.
    file_overrides: A path to a JSON file with hyperparameter overrides.

  Returns:
    Experiment hyperparmeters with updated values.
  """
  hyperparameters_overrides_manual = (
      json.loads(manual_overrides) if manual_overrides else None
  )
  if file_overrides:
    with tf.io.gfile.GFile(file_overrides, "r") as f:
      hyperparameters_overrides_from_file = json.load(f)
  else:
    hyperparameters_overrides_from_file = None
  aggregated_hyperparameters_overrides = tuple(
      itertools.filterfalse(
          lambda overrides: not overrides,
          [
              hyperparameters_overrides_manual,
              hyperparameters_overrides_from_file,
          ],
      )
  )
  hyperparameters_overrides = _merge_hyperparameters(
      hyperparameters=aggregated_hyperparameters_overrides
  )
  for key in _NESTED_HYPERPARAMETERS:
    if key in hyperparameters_overrides:
      hyperparameters[key] = {}
  hyperparameters.update(hyperparameters_overrides)
  return hyperparameters


def build_hyperparameters(
    *,
    actions_hyperparameters: config_dict.ConfigDict,
    agent_hyperparameters: config_dict.ConfigDict,
    reward_hyperparameters: config_dict.ConfigDict,
    data_pipeline_hyperparameters: config_dict.ConfigDict,
    trainer_hyperparameters: config_dict.ConfigDict,
    checkpoint_directory: str,
    artifact_directory: str,
    hyperparameters_file: str | None = None,
    hyperparameters_overrides: str | None = None,
) -> config_dict.ConfigDict:
  """Assembles all the experiment hyperparameters.

  Args:
    actions_hyperparameters: A mapping between action class hyperparameters and
      values.
    agent_hyperparameters: A mapping between agent class hyperparameters and
      values.
    reward_hyperparameters: A mapping between reward class hyperparameters and
      values.
    data_pipeline_hyperparameters: A mapping between data_pipeline class
      hyperparameters and values.
    trainer_hyperparameters: A mapping between trainer class hyperparameters and
      values.
    checkpoint_directory: A path to the directory where model checkpoints
      will be saved.
    artifact_directory: A path to the directory where all the training artifacts
      (e.g. TensorBoard logs) will be saved.
    hyperparameters_file: A path to the file with hyperparameters overrides.
    hyperparameters_overrides: A JSON string with hyperparameters overrides.

  Returns:
    Assembled experiment hyperparameters.
  """
  core_hyperparameters = _aggregate_core_hyperparameters(
      actions_hyperparameters=actions_hyperparameters,
      agent_hyperparameters=agent_hyperparameters,
      reward_hyperparameters=reward_hyperparameters,
      data_pipeline_hyperparameters=data_pipeline_hyperparameters,
      trainer_hyperparameters=trainer_hyperparameters,
  )
  experiment_hyperparameters = core_hyperparameters + (
      {
          "checkpoint_directory": checkpoint_directory,
          "artifact_directory": artifact_directory,
      },
  )
  experiment_hyperparameters = _merge_hyperparameters(
      hyperparameters=experiment_hyperparameters
  )
  experiment_hyperparameters = _convert_nested_hyperparameters(
      hyperparameters=experiment_hyperparameters
  )
  experiment_hyperparameters = _update_hyperparameters_from_overrides(
      hyperparameters=experiment_hyperparameters,
      manual_overrides=hyperparameters_overrides,
      file_overrides=hyperparameters_file,
  )
  experiment_hyperparameters = _verify_action_hyperparameters(
      hyperparameters=experiment_hyperparameters
  )
  return _verify_trainer_hyperparameters(
      hyperparameters=experiment_hyperparameters
  )


def _get_model_data_type_from_string(*, model_data_type: str) -> jnp.dtype:
  """Maps a string with a jax.numpy dtype."""
  if model_data_type not in ("float32", "float64"):
    raise ValueError(f"Invalid dtype: {model_data_type}")
  return jnp.dtype(model_data_type)


def set_hyperparameters(
    *,
    actions_hyperparameters: config_dict.ConfigDict,
    agent_hyperparameters: config_dict.ConfigDict,
    reward_hyperparameters: config_dict.ConfigDict,
    data_pipeline_hyperparameters: config_dict.ConfigDict,
    trainer_hyperparameters: config_dict.ConfigDict,
    checkpoint_directory: str,
    artifact_directory: str,
    hyperparameters_file: str | None = None,
    hyperparameters_overrides: str | None = None,
) -> config_dict.ConfigDict:
  """Assembles all the experiment hyperparameters and saves them in a JSON file.

  Args:
    actions_hyperparameters: A mapping between action class hyperparameters and
      values.
    agent_hyperparameters: A mapping between agent class hyperparameters and
      values.
    reward_hyperparameters: A mapping between reward class hyperparameters and
      values.
    data_pipeline_hyperparameters: A mapping between data pipeline class
      hyperparameters and values.
    trainer_hyperparameters: A mapping between trainer class hyperparameters and
      values.
    checkpoint_directory: A path to the directory where model checkpoints
      will be saved.
    artifact_directory: A path to the directory where all the training artifacts
      (e.g. TensorBoard logs) will be saved.
    hyperparameters_file: A path to the file with hyperparameters overrides.
    hyperparameters_overrides: A JSON string with hyperparameters overrides.

  Returns:
    Assembled experiment hyperparameters.
  """
  hyperparameters = build_hyperparameters(
      actions_hyperparameters=actions_hyperparameters,
      agent_hyperparameters=agent_hyperparameters,
      reward_hyperparameters=reward_hyperparameters,
      data_pipeline_hyperparameters=data_pipeline_hyperparameters,
      trainer_hyperparameters=trainer_hyperparameters,
      checkpoint_directory=checkpoint_directory,
      artifact_directory=artifact_directory,
      hyperparameters_file=hyperparameters_file,
      hyperparameters_overrides=hyperparameters_overrides,
  )
  logging.info("Experiment hyperparameters are: %s", hyperparameters)
  hyperparameters_path = os.path.join(
      hyperparameters.artifact_directory, "hyperparameters.json"
  )
  with tf.io.gfile.GFile(hyperparameters_path, "w") as f:
    json.dump(hyperparameters.to_dict(), f)
    logging.info(
        "A JSON file with experiment hyperparameters saved as %s",
        hyperparameters_path,
    )
  if hyperparameters.model_data_type:
    converted_model_dtype = _get_model_data_type_from_string(
        model_data_type=hyperparameters.model_data_type
    )
    with hyperparameters.unlocked():
      del hyperparameters["model_data_type"]
      hyperparameters["model_data_type"] = converted_model_dtype
  return hyperparameters


def set_hardware(
    *,
    accelerator: str = "cpu",
    coordinator_address: str | None = None,
    process_number: int = 1,
    process_id: int = 0,
):
  """Sets up hardaware for the training loop.

  Args:
    accelerator: Machine accelerator type.
    coordinator_address: The IP address of process `0` and a port on which that
      process should launch a coordinator service.
    process_number: Number of processes.
    process_id: The ID number of the current process.

  Raises:
    ValueError: An error when using a GPU but not providing any coordinator
      address. Or when providing an incompatible accelerator name.
  """
  tf.config.experimental.set_visible_devices([], "GPU")
  logging.info(
      "Hid any GPU devices from Tensorflow since we only use it for data"
      " preprocessing."
  )
  if accelerator not in _VALID_ACCELERATORS:
    raise ValueError(
        f"Accelerator must be one of: {_VALID_ACCELERATORS}. "
        f"Received: {accelerator!r}"
    )
  if accelerator == "tpu":
    jax.distributed.initialize()
  elif accelerator == "gpu":
    if not coordinator_address:
      raise ValueError("coordinator_address is required when you use a GPU.")
    jax.distributed.initialize(
        coordinator_address, num_processes=process_number, process_id=process_id
    )
  elif accelerator == "cpu":
    port = portpicker.pick_unused_port()
    jax.distributed.initialize(
        f"localhost:{port}", num_processes=process_number, process_id=process_id
    )
  logging.info(
      "Initialized JAX distributed for a single-process setting.",
  )
  logging.info(
      "There are %s accelerator devices in the cluster visible to JAX.",
      jax.device_count(),
  )
  logging.info(
      (
          "There are %s accelerator devices attached to this host and visible"
          " to JAX."
      ),
      jax.local_device_count(),
  )
