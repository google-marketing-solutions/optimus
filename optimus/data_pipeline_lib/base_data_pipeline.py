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

"""Base class for all data pipelines."""
import abc
import functools
from typing import Callable, Final
from absl import logging
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf

AUTOTUNE = tf.data.experimental.AUTOTUNE

DEFAULT_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        input_dimensions=config_dict.placeholder(int),
        categorical_dimensions=config_dict.placeholder(list),
        categorical_indexes=config_dict.placeholder(list),
        shuffle_size=512,
        batch_size=1024,
        evaluation_batch_size=8,
        train_dataset_size=config_dict.placeholder(int),
        data_pipeline_name="base_data_pipeline",
        reactions_dimensions=1,
    )
)

_TRAIN: Final[str] = "train"
_PRETRAIN: Final[str] = "pretrain"
_EVALUATION: Final[str] = "evaluation"
_CalculateRewardFunction = Callable[[tf.Tensor, tf.Tensor, bool], tf.Tensor]


def process_data(
    *,
    tensor: tf.Tensor,
    split: str = "train",
    hyperparameters: config_dict.ConfigDict,
    reward_calculation_function: _CalculateRewardFunction | None = None,
) -> dict[str, tf.Tensor]:
  """Preprocesses tensors in either train or evaluation format.

  Args:
    tensor: A tensor from a TF Dataset.
    split: A split of data to preprocess.
    hyperparameters: A data pipeline class hyperparameters.
    reward_calculation_function: A function to calculate rewards based on taken
      action and recorded user reactions.

  Returns:
    A dictionary of tensors in either train or evaluation format.

  Raises:
    ValueError: An error when split not equal to "train" or "evaluation".
  """
  if split == _TRAIN:
    (
        states_tensor,
        actions_tensor,
        values_tensor,
        log_probabilities_tensor,
        dones_tensor,
        attentive_transformer_losses_tensor,
        reactions_tensor,
    ) = tf.split(
        tensor,
        [
            hyperparameters.input_dimensions,
            1,
            1,
            1,
            1,
            1,
            hyperparameters.reactions_dimensions,
        ],
    )
    rewards_tensor = reward_calculation_function(
        actions_tensor,
        reactions_tensor,
        hyperparameters.sign_rewards,
    )
    observation = dict(
        states=states_tensor,
        actions=actions_tensor,
        rewards=rewards_tensor,
        values=values_tensor,
        log_probabilities=log_probabilities_tensor,
        dones=dones_tensor,
        attentive_transformer_losses=attentive_transformer_losses_tensor,
    )
  elif split == _EVALUATION:
    states_tensor, target_actions_tensor = tf.split(
        tensor, [hyperparameters.input_dimensions, 1]
    )
    observation = dict(
        states=states_tensor,
        target_actions=target_actions_tensor,
    )
  else:
    raise ValueError(
        f"Split must be either 'train' or 'evaluation', you passed: {split!r}"
    )
  return observation


def calculate_train_and_evaluation_batch_size(
    *,
    process_count: int,
    number_of_devices: int,
    hyperparameters: config_dict.ConfigDict,
) -> tuple[int, int]:
  """Verifies and calculates train and evaluation batch sizes.

  Args:
    process_count: A number of processes detected by JAX.
    number_of_devices: A number of devices (e.g. TPUs) detected by JAX.
    hyperparameters: A data pipeline class hyperparameters.

  Returns:
    Train batch size per device and evaluation batch size per device.

  Raises:
    ValueError: An error when: 1) batch_size isn't divisible by the number of
    detected processes, 2) train_dataset_size / batch_size is less than
    train_steps, 3) evaluation_batch_size isn't divisible by the number of
    detected processes, 4) per_host_batch_size isn't divisible by the number of
    detected devices or 5) per_host_evaluation_batch_size isn't divisible by the
    number of detected devices.
  """
  if hyperparameters.batch_size % process_count:
    raise ValueError(
        f"process_count={process_count} must divide"
        f" batch_size={hyperparameters.batch_size}."
    )
  per_host_batch_size = hyperparameters.batch_size // process_count
  if per_host_batch_size % number_of_devices != 0:
    raise ValueError(
        f"number_of_devices={number_of_devices} must divide"
        f" per_host_batch_size={per_host_batch_size}."
    )
  if (
      hyperparameters.train_dataset_size / per_host_batch_size
      < hyperparameters.train_steps
  ):
    raise ValueError(
        f"train_dataset_size={hyperparameters.train_dataset_size} divided"
        f" by per_host_batch_size={per_host_batch_size} must be"
        f" more than train_steps={hyperparameters.train_steps}."
    )
  evaluation_batch_size = (
      hyperparameters.evaluation_batch_size or hyperparameters.batch_size
  )
  if evaluation_batch_size % process_count:
    raise ValueError(
        f"process_count={process_count} must divide"
        f" evaluation_batch_size={evaluation_batch_size}."
    )
  per_host_evaluation_batch_size = evaluation_batch_size // process_count
  if per_host_evaluation_batch_size % number_of_devices != 0:
    raise ValueError(
        f"number_of_devices={number_of_devices} must divide"
        f" per_host_evaluation_batch_size={per_host_evaluation_batch_size}."
    )
  return (
      per_host_batch_size,
      per_host_evaluation_batch_size,
  )


class BaseDataPipeline(metaclass=abc.ABCMeta):
  """Abstract parent class for all data pipeline classes.

  Attributes:
    hyperparameters: Training hyperparameters.
    per_host_batch_sizes: A sequence with per host train and evaluation batch
      sizes.
  """

  def __init__(
      self,
      *,
      hyperparameters: config_dict.ConfigDict,
  ) -> None:
    """Initializes the BaseDataPipeline class.

    Args:
      hyperparameters: The hyperparameteres for the data pipeline class.
    """
    self.hyperparameters = hyperparameters
    self._dataset_rng_key = jax.random.PRNGKey(
        hyperparameters.training_rng_seed
    )

  def build_tensorflow_pipeline(
      self,
      *,
      data: jnp.ndarray | np.ndarray,
      split: str = "train",
      batch_size: int = 1024,
      drop_remainder: bool = True,
      reward_calculation_function: _CalculateRewardFunction | None = None,
  ) -> tf.data.Dataset:
    """Returns a TensorFlow data pipeline.

    Args:
      data: Either train or evaluation data.
      split: A split of data to preprocess.
      batch_size: A batch size in the pipeline.
      drop_remainder: An indicator whether to drop remainder during batching.
      reward_calculation_function: A function to calculate rewards based on
        taken action and recorded user reactions.

    Raises:
      ValueError: An error when split not equal to "train" or "evaluation".
    """
    dataset = tf.data.Dataset.from_tensor_slices(data)
    if split == _TRAIN or split == _PRETRAIN:
      dataset = dataset.shuffle(
          self.hyperparameters.shuffle_size,
          seed=jax.random.bits(self._dataset_rng_key).item(),
      )
    elif split == _EVALUATION:
      pass
    else:
      raise ValueError(
          "Split must be either 'train', 'pretrain' or 'evaluation', you"
          f" passed: {split}"
      )
    if split != _PRETRAIN:
      partial_process_data = functools.partial(
          process_data,
          split=split,
          hyperparameters=self.hyperparameters,
          reward_calculation_function=reward_calculation_function,
      )
      dataset = dataset.map(lambda x: partial_process_data(tensor=x), AUTOTUNE)
    dataset = dataset.batch(
        batch_size=batch_size, drop_remainder=drop_remainder
    )
    return dataset.prefetch(AUTOTUNE)

  @functools.cached_property
  def per_host_batch_sizes(self) -> tuple[int, int]:
    """Returns train and evaluation per host batch sizes."""
    process_count = jax.process_count()
    logging.info("JAX counted %s process(es).", process_count)
    number_of_devices = jax.local_device_count()
    logging.info("JAX counted %s local device(s).", number_of_devices)
    per_host_batch_size, per_host_evaluation_batch_size = (
        calculate_train_and_evaluation_batch_size(
            process_count=process_count,
            number_of_devices=number_of_devices,
            hyperparameters=self.hyperparameters,
        )
    )
    logging.info(
        "The calculated per_host_batch_size is : %s", per_host_batch_size
    )
    logging.info(
        "The calculated per_host_evaluation_batch_size is : %s",
        per_host_evaluation_batch_size,
    )
    return per_host_batch_size, per_host_evaluation_batch_size

  def train_data_pipeline(
      self,
      *,
      train_data: jnp.ndarray | np.ndarray,
      reward_calculation_function: _CalculateRewardFunction,
  ) -> tf.data.Dataset:
    """Returns a train TensorFlow data pipeline.

    Args:
      train_data: Train data.
      reward_calculation_function: A TensorFlow function to calculate rewards
        from "actions" and "reactions".
    """
    return self.build_tensorflow_pipeline(
        data=train_data,
        split=_TRAIN,
        batch_size=self.per_host_batch_sizes[0],
        drop_remainder=True,
        reward_calculation_function=reward_calculation_function,
    )

  def pretrain_data_pipeline(
      self,
      *,
      pretrain_data: jnp.ndarray | np.ndarray,
  ) -> tf.data.Dataset:
    """Returns a pretrain TensorFlow data pipeline.

    Args:
      pretrain_data: Pretrain data.
    """
    return self.build_tensorflow_pipeline(
        data=pretrain_data,
        split=_PRETRAIN,
        batch_size=self.per_host_batch_sizes[0],
        drop_remainder=True,
    )

  def evaluation_data_pipeline(
      self, *, evaluation_data: jnp.ndarray | np.ndarray
  ) -> tf.data.Dataset:
    """Returns an evaluation TensorFlow data pipeline.

    Args:
      evaluation_data: Evaluation data.
    """
    return self.build_tensorflow_pipeline(
        data=evaluation_data,
        split=_EVALUATION,
        batch_size=self.per_host_batch_sizes[1],
        drop_remainder=False,
    )
