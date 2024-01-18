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

"""Base class for all trainers."""

import abc
import dataclasses
import functools
import itertools
from typing import Any, Callable, Mapping

from absl import logging
import flax
from flax.core import frozen_dict
import flax.linen as nn
from flax.metrics import tensorboard
from flax.training import common_utils
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
import optax
import orbax.checkpoint
import tensorflow as tf
import tqdm

from optimus.agent_lib import base_agent
from optimus.data_pipeline_lib import data_pipelines
from optimus.reward_lib import base_reward
from optimus.trainer_lib import losses
from optimus.trainer_lib import optimizers
from optimus.trainer_lib import schedules

DEFAULT_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        train_steps=config_dict.placeholder(int),
        number_of_epochs=1,
        evaluation_frequency=1,
        evaluation_steps=1,
        gae_gamma=0.99,
        gae_lambda=0.95,
        clip_parameters=True,
        clip_parameters_coefficient=0.1,
        value_function_coefficient=0.5,
        entropy_coefficient=0.02,
        lambda_sparse=1e-3,
        training_rng_seed=-1,
        loss="tabnet_proximal_policy_optimization_loss",
        learning_rate_hyperparameters={
            "initial_value": 0.01,
            "schedule": "linear_schedule",
            "end_value": 0.0,
            "transition_steps": config_dict.placeholder(
                int
            ),  # Advised to be equal to train_steps
        },
        optimizer="adam",
        optimizer_hyperparameters={
            "b1": 0.9,
            "b2": 0.999,
            "eps": 1e-8,
            "eps_root": 0.0,
        },
        exploration_exploitation_hyperparameters={
            "schedule": "constant_schedule",
            "initial_value": 0.0,
        },
        checkpoint_frequency=1,
        checkpoints_kept=100,
        trainer_name="base_trainer",
    )
)


def calculate_advantages(
    previous_loop_result: float,
    current_array_element: jnp.ndarray,
    gae_gamma: float,
    gae_lambda: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
  """Calculates Generalized Advantage Estimation (GAE) advantages.

  See: https://arxiv.org/abs/1707.06347

  Args:
    previous_loop_result: The previous GAE value.
    current_array_element: The current element of stacked rewards, dones (masks)
      and values arrays.
    gae_gamma: A GAE gamma discounting factor.
    gae_lambda: A GAE lambda regularizing factor.

  Returns:
    An array with GAE advantages.

  Raises:
    ValueError: An error when length of the value array is too short.
  """
  reward, mask, value, next_value = current_array_element
  value_difference = gae_gamma * next_value * mask - value
  delta = reward + value_difference
  previous_loop_result = (
      delta
      + gae_gamma * gae_lambda * current_array_element[1] * previous_loop_result
  )
  return previous_loop_result, previous_loop_result


def calculate_trajectories(
    *, batch: Any, hyperparameters: config_dict.ConfigDict
) -> jnp.ndarray:
  """Processes agent's experience to calculate trajectories.

  Args:
    batch: A batch of data to be processed.
    hyperparameters: Experiment hyperparameters.

  Returns:
    A processed agent's experience described as trajectories.
  """
  values = batch["values"]
  modified_values = jnp.vstack((values, values[-1])).flatten()
  terminal_masks = batch["dones"].flatten()
  processed_terminal_masks = jnp.where(
      terminal_masks == 1.0, terminal_masks == 0.0, terminal_masks
  )
  stacked_arrays = jnp.stack(
      (
          batch["rewards"].flatten(),
          processed_terminal_masks,
          modified_values[:-1],
          modified_values[1:],
      ),
      axis=1,
  )
  partial_calculate_advantages = functools.partial(
      calculate_advantages,
      gae_gamma=hyperparameters.gae_gamma,
      gae_lambda=hyperparameters.gae_lambda,
  )
  generalized_advantage_estimation = 0.0
  _, unprocessed_advantages = jax.lax.scan(
      partial_calculate_advantages,
      generalized_advantage_estimation,
      stacked_arrays,
  )
  advantages = unprocessed_advantages.reshape(batch["states"].shape[0], 1)
  return jnp.concatenate(
      [
          batch["states"],
          batch["actions"],
          batch["log_probabilities"],
          advantages + values,
          advantages,
          batch["attentive_transformer_losses"],
      ],
      axis=1,
  )


def calculate_train_step_metrics(
    *,
    step: int,
    learning_rate: Callable[[int], float],
    exploration_exploitation_rate: Callable[[int], float],
    hyperparameters: config_dict.ConfigDict,
) -> dict[str, float]:
  """Returns a mapping between train step metrics and their values.

  Args:
    step: The current train step.
    learning_rate: A learning rate schedule.
    exploration_exploitation_rate: A exploration_exploitation_rate schedule.
    hyperparameters: Experiment hyperparameters.
  """
  alpha = (
      1.0 - step / hyperparameters.train_steps
      if hyperparameters.clip_parameters
      else 1.0
  )
  clip_parameters_coefficient = (
      hyperparameters.clip_parameters_coefficient * alpha
  )
  return {
      "step_learning_rate": learning_rate(step),
      "step_clip_parameters_coefficient": clip_parameters_coefficient,
      "step_exploration_exploitation_rate": exploration_exploitation_rate(step),
  }


def train_step(
    model_state: base_agent.BaseAgentState,
    *,
    exploration_exploitation_rate: float,
    clip_parameters_coefficient: float,
    batch: jnp.ndarray,
    loss_function: float | Callable[[float], float],
    hyperparameters: config_dict.ConfigDict,
) -> tuple[base_agent.BaseAgentState, jnp.ndarray]:
  """Returns an updated model state after a single training step.

  Args:
    model_state: The current model state.
    exploration_exploitation_rate: A value that controls how often an agent will
      choose an unoptimal action to explore.
    clip_parameters_coefficient: The clip parameters coefficient for a single
      step.
    batch: An input train batch.
    loss_function: The model's loss function, i.e.train_cost.
    hyperparameters: Experiment hyperparameters.
  """
  grad_function = jax.value_and_grad(loss_function, has_aux=True)
  (loss, updates), gradients = grad_function(
      model_state.params,
      model_state.batch_stats,
      model_state.apply_fn,
      batch=batch,
      clip_parameters_coefficient=clip_parameters_coefficient,
      hyperparameters=hyperparameters,
  )
  model_state = model_state.apply_gradients(
      grads=jax.lax.pmean(gradients, "batch")
  )
  model_state = model_state.replace(
      batch_stats=updates["batch_stats"],
      exploration_exploitation_epsilon=exploration_exploitation_rate,
  )
  loss = jax.lax.pmean(loss, "batch")
  return model_state, loss


def _sync_batch_statistics(
    *, model_state: base_agent.BaseAgentState
) -> base_agent.BaseAgentState:
  """Returns a model state with synced batch statistics across replicas.

  Args:
    model_state: The current model state.
  """
  cross_replica_mean = jax.pmap(lambda x: jax.lax.pmean(x, "x"), "x")
  return model_state.replace(
      batch_stats=cross_replica_mean(model_state.batch_stats)
  )


def evaluate(
    *,
    model_state: base_agent.BaseAgentState,
    evaluation_data: tf.data.Dataset,
    pmapped_evaluation_step: Any,
    evaluation_steps: int,
) -> dict[str, float]:
  """Calculates an evaluation reward.

  Args:
    model_state: The current model state.
    evaluation_data: A TensorFlow evaluation data pipeline.
    pmapped_evaluation_step: A pmapped evaluation step.
    evaluation_steps: A number of evaluation steps.

  Returns:
    A total evaluation reward.
  """
  evaluation_metrics = []
  for evaluation_batch in itertools.islice(
      evaluation_data, 0, evaluation_steps
  ):
    converted_evaluation_batch = jax.tree_util.tree_map(
        jnp.asarray, evaluation_batch
    )
    concatenated_evaluation_batch = jnp.concatenate(
        [
            converted_evaluation_batch["states"],
            converted_evaluation_batch["target_actions"],
        ],
        axis=1,
    )
    sharded_evaluation_batch = common_utils.shard(concatenated_evaluation_batch)
    evaluation_reward = pmapped_evaluation_step(
        agent_state=model_state,
        batch=sharded_evaluation_batch,
    )
    evaluation_metrics.append(
        dict(evaluation_reward=evaluation_reward.sum(keepdims=True))
    )
  return jax.tree_util.tree_map(
      jnp.sum, common_utils.get_metrics(evaluation_metrics)
  )


def evaluation_step(
    *,
    agent_state: base_agent.BaseAgentState,
    batch: jnp.ndarray,
    input_dimensions: int,
    agent: base_agent.BaseAgent,
    reward: base_reward.BaseReward,
) -> jnp.ndarray:
  """Calculates evaluation rewards given the selected and target actions.

  Args:
    agent_state: Current agent state.
    batch: An input batch with context for the agent to make predictions and
      target_actions to calculate evaluation rewards.
    input_dimensions: The number of dimesions in the context state.
    agent: A training agent.
    reward: A training reward.

  Returns:
    Evaluation rewards and booleans indicating if a sequence is complete.
  """
  states, target_actions = jnp.array_split(batch, [input_dimensions], axis=1)
  predicted_actions = agent.evaluate_predict(
      agent_state=agent_state, batch=states
  )
  return reward.calculate_evaluation_reward(
      predicted_actions=predicted_actions,
      target_actions=target_actions,
  )


def initialize_evaluation_data(
    *,
    hyperparameters: config_dict.ConfigDict,
    evaluation_dataset: tf.data.Dataset | None = None,
) -> tf.data.Dataset | None:
  """Returns an evaluation data pipeline.

  If BaseTrainer has evaluation_dataset loaded then evaluation_frequency
  and evaluation_steps hyperparameters are required for this method to work.

  Args:
    hyperparameters: Training hyperparameters.
    evaluation_dataset: A dataset with evaluation data with tensors of "states"
      and "dones".

  Raises:
    ValueError: An error if the constraints described above are violated.
  """
  if not evaluation_dataset:
    return None
  requirement_one = (
      "evaluation_frequency" in hyperparameters
      and hyperparameters.evaluation_frequency
  )
  requirement_two = (
      "evaluation_steps" in hyperparameters and hyperparameters.evaluation_steps
  )
  if not (requirement_one and requirement_two):
    raise ValueError(
        "The evaluation_frequency and evaluation_steps hyperparameters are "
        "required when evaluation_dataset is provided."
    )
  return evaluation_dataset


def initialize_model(
    *,
    initialization_rng_key: jax.Array,
    agent: base_agent.BaseAgent,
) -> tuple[nn.Module, frozen_dict.FrozenDict | Mapping[str, Any]]:
  """Initializes a Flax model.

  Args:
    initialization_rng_key: An RNG key for model initialization.
    agent: A training agent.

  Returns:
    Initialized FLAX model and initial model parameters.
  """
  dummy_initialization_batch = agent.get_dummy_inputs()
  flax_model = agent.flax_module
  initial_parameters = jax.jit(flax_model.init, static_argnames="train")(
      rngs={"params": initialization_rng_key},
      input_x=dummy_initialization_batch,
      train=False,
  )
  logging.info("Finished initializing the model.")
  logging.info("Model parameters: %s", initial_parameters)
  parameter_count = sum(
      x.size for x in jax.tree_util.tree_leaves(initial_parameters["params"])
  )
  logging.info("Number of model parameters: %s", parameter_count)
  return flax_model, initial_parameters


def initialize_schedule(
    *, hyperparameters: config_dict.ConfigDict
) -> Callable[[int], float] | float:
  """Returns a configured schedule.

  Args:
    hyperparameters: The schedule hyperparameters.
  """
  return schedules.get_schedule(schedule_name=hyperparameters.schedule)(
      hyperparameters
  )


def initialize_tensorboard_writer(
    *,
    hyperparameters: config_dict.ConfigDict,
) -> tensorboard.SummaryWriter:
  """Returns a TensorBoard SummaryWriter.

  Args:
    hyperparameters: Training hyperparameters.
  """
  tensorboard_writer = tensorboard.SummaryWriter(
      hyperparameters.artifact_directory
  )
  tensorboard_writer.hparams(hyperparameters.to_dict())
  return tensorboard_writer


def initialize_checkpointing(
    *,
    hyperparameters: config_dict.ConfigDict,
) -> tuple[orbax.checkpoint.CheckpointManager, int]:
  """Returns an Orbax CheckpointManager and the latest training step.

  Args:
    hyperparameters: Training hyperparameters.
  """
  checkpoint_manager_options = orbax.checkpoint.CheckpointManagerOptions(
      save_interval_steps=hyperparameters.checkpoint_frequency,
      max_to_keep=hyperparameters.checkpoints_kept,
      step_prefix="checkpoint",
      save_on_steps=(hyperparameters.train_steps,),
  )
  async_checkpointer = orbax.checkpoint.AsyncCheckpointer(
      orbax.checkpoint.PyTreeCheckpointHandler()
  )
  checkpoint_manager = orbax.checkpoint.CheckpointManager(
      directory=hyperparameters.checkpoint_directory,
      checkpointers=async_checkpointer,
      options=checkpoint_manager_options,
  )
  latest_training_step = checkpoint_manager.latest_step()
  if not latest_training_step:
    logging.info(
        "No checkpoint found under %s.",
        hyperparameters.checkpoint_directory,
    )
    latest_training_step = 0
  else:
    logging.info(
        "The latest saved checkpoint is: %s. The training loop will start"
        " from it.",
        latest_training_step,
    )
  logging.info("Training checkpointing is set up.")
  return checkpoint_manager, latest_training_step


def restore_model_state(
    *,
    checkpoint_manager: orbax.checkpoint.CheckpointManager,
    model_state: base_agent.BaseAgentState,
    hyperparameters: config_dict.ConfigDict,
    transforms: Any | None = None,
    mesh: jax.sharding.Mesh | None = None,
) -> base_agent.BaseAgentState:
  """Returns a restored TrainState.

  Args:
    checkpoint_manager: An Orbax CheckpointManager for the training procedure.
    model_state: A custom model state from the already initialized model.
    hyperparameters: Training hyperparameters.
    transforms: Transformations to apply to checkpoint found in `directory` (see
      orbax.checkpoint.transform_utils).
    mesh: Device mesh.
  """

  def _restore_arguments():
    if mesh is not None:
      return orbax.checkpoint.ArrayRestoreArgs(
          restore_type=jax.Array,
          mesh=mesh,
          mesh_axes=jax.sharding.PartitionSpec(),
      )
    return orbax.checkpoint.RestoreArgs(restore_type=np.ndarray)

  restore_arguments = jax.tree_util.tree_map(
      lambda _: _restore_arguments(), model_state
  )
  if transforms is None:
    transforms = {}
  transforms["step"] = orbax.checkpoint.Transform()
  restore_keyword_arguments = dict(
      restore_args=restore_arguments, transforms=transforms
  )
  return checkpoint_manager.restore(
      step=checkpoint_manager.latest_step(),
      restore_kwargs=restore_keyword_arguments,
      items=model_state,
      directory=hyperparameters.checkpoint_directory,
  )


def pmap_train_and_evaluation(
    *,
    agent: base_agent.BaseAgent,
    reward: base_reward.BaseReward,
    hyperparameters: config_dict.ConfigDict,
) -> tuple[Any, Any]:
  """Pmaps train and evaluation functions to allow for multi-device training.

  Args:
    agent: A training agent.
    reward: A training reward.
    hyperparameters: Training hyperparameters.

  Returns:
    Pmapped train and evaluation functions.
  """
  pmapped_train_step = jax.pmap(
      functools.partial(
          train_step,
          hyperparameters=hyperparameters,
          loss_function=losses.get_loss(loss_name=hyperparameters.loss),
      ),
      axis_name="batch",
  )
  pmapped_evaluation_step = jax.pmap(
      functools.partial(
          evaluation_step,
          input_dimensions=hyperparameters.input_dimensions,
          agent=agent,
          reward=reward,
      ),
      axis_name="batch",
  )
  logging.info("Training and evaluation functions are pmapped.")
  return pmapped_train_step, pmapped_evaluation_step


def is_evaluation_step(
    *, step: int, hyperparameters: config_dict.ConfigDict
) -> bool:
  """Returns an indicator whether to calculate evaluation metrics summary.

  Args:
    step: The latest training step from the Orbax Checkpoint Manager.
    hyperparameters: Training hyperparameters.
  """
  final_step = step == hyperparameters.train_steps - 1
  evaluation_metrics_summary_step = (
      step % hyperparameters.evaluation_frequency == 0
  )
  return final_step or evaluation_metrics_summary_step


def _process_replicated_model_state(
    *,
    model_state: base_agent.BaseAgentState,
) -> base_agent.BaseAgentState:
  """Returns an unreplicate model state with synced batch statistics.

  Args:
    model_state: The current model state.
  """
  model_state = _sync_batch_statistics(model_state=model_state)
  return jax.device_get(jax.tree_util.tree_map(lambda x: x[0], model_state))


def initialize_model_state(
    *,
    model: nn.Module,
    initial_parameters: frozen_dict.FrozenDict | Mapping[str, Any],
    optimizer: optax.GradientTransformation,
    exploration_exploitation_epsilon: float,
) -> base_agent.BaseAgentState:
  """Creates a custom model state.

  Args:
    model: An initialized Flax model.
    initial_parameters: Model parameters from an initialized model.
    optimizer: An initialized optimizer.
    exploration_exploitation_epsilon: A value that controls how often an agent
      will choose and an unoptimal action to explore.

  Returns:
    Custom model state.
  """
  model_state = base_agent.BaseAgentState.create(
      apply_fn=model.apply,
      params=initial_parameters["params"],
      tx=optimizer,
      batch_stats=initial_parameters["batch_stats"],
      exploration_exploitation_epsilon=exploration_exploitation_epsilon,
  )
  logging.info("Finished creating the model state.")
  return model_state


def initialize_model_state_for_prediction(
    *,
    agent: base_agent.BaseAgent,
    hyperparameters: config_dict.ConfigDict,
) -> base_agent.BaseAgentState:
  """Returns a model state initialized for prediction.

  Args:
    agent: A training agent.
    hyperparameters: Training hyperparameters.
  """
  rng_key = jax.random.PRNGKey(hyperparameters.training_rng_seed)
  initialization_rng_key, _ = jax.random.split(rng_key, 2)
  model, initial_parameters = initialize_model(
      initialization_rng_key=initialization_rng_key,
      agent=agent,
  )
  learning_rate = initialize_schedule(
      hyperparameters=hyperparameters.learning_rate_hyperparameters
  )
  exploration_exploitation_rate = initialize_schedule(
      hyperparameters=hyperparameters.exploration_exploitation_hyperparameters
  )
  optimizer = optimizers.get_optimizer(
      optimizer_name=hyperparameters.optimizer
  )(learning_rate, **hyperparameters.optimizer_hyperparameters)
  checkpoint_manager, train_loop_start_step = initialize_checkpointing(
      hyperparameters=hyperparameters
  )
  model_state = initialize_model_state(
      model=model,
      initial_parameters=initial_parameters,
      optimizer=optimizer,
      exploration_exploitation_epsilon=exploration_exploitation_rate(
          train_loop_start_step
      ),
  )
  if train_loop_start_step != 0:
    model_state = restore_model_state(
        checkpoint_manager=checkpoint_manager,
        model_state=model_state,
        hyperparameters=hyperparameters,
    )
  return model_state


@dataclasses.dataclass(frozen=True)
class TrainingBuildingBlocks:
  """Instance with and an agent class name, its instance and hyperparameters.

  Attributes:
    train_rng_key: A PRNG key for training repeatability.
    learning_rate: A learning rate schedule.
    exploration_exploitation_rate: A value that controls how often an agent will
      choose and an unoptimal action to explore.
    tensorboard_writer: The TensorBoard Summary Writer of the training
      procedure.
    train_loop_start_step: The latest training step from the Orbax Checkpoint
      Manager.
    checkpoint_manager: An Orbax CheckpointManager for the training procedure.
    pmapped_train_step: A pmapped train function.
    pmapped_evaluation_step: A pmapped evaluation function.
    model_state: An unreplicated model state.
    replicated_model_state: A replicated model state.
  """

  train_rng_key: jax.Array
  learning_rate: Callable[[int], float] | float
  exploration_exploitation_rate: Callable[[int], float] | float
  tensorboard_writer: tensorboard.SummaryWriter
  train_loop_start_step: int
  checkpoint_manager: orbax.checkpoint.CheckpointManager
  pmapped_train_step: Any
  pmapped_evaluation_step: Any
  model_state: base_agent.BaseAgentState
  replicated_model_state: base_agent.BaseAgentState

  @classmethod
  def prepare(
      cls,
      *,
      agent: base_agent.BaseAgent,
      reward: base_reward.BaseReward,
      hyperparameters: config_dict.ConfigDict,
  ) -> "TrainingBuildingBlocks":
    """Returns building blocks necessary for the training procedure.

    Args:
      agent: A training agent.
      reward: A training reward.
      hyperparameters: Training hyperparameters.
    """
    rng_key = jax.random.PRNGKey(hyperparameters.training_rng_seed)
    initialization_rng_key, train_rng_key = jax.random.split(rng_key, 2)
    model, initial_parameters = initialize_model(
        initialization_rng_key=initialization_rng_key,
        agent=agent,
    )
    learning_rate = initialize_schedule(
        hyperparameters=hyperparameters.learning_rate_hyperparameters
    )
    exploration_exploitation_rate = initialize_schedule(
        hyperparameters=hyperparameters.exploration_exploitation_hyperparameters
    )
    optimizer = optimizers.get_optimizer(
        optimizer_name=hyperparameters.optimizer
    )(learning_rate, **hyperparameters.optimizer_hyperparameters)
    tensorboard_writer = initialize_tensorboard_writer(
        hyperparameters=hyperparameters
    )
    checkpoint_manager, train_loop_start_step = initialize_checkpointing(
        hyperparameters=hyperparameters
    )
    model_state = initialize_model_state(
        model=model,
        initial_parameters=initial_parameters,
        optimizer=optimizer,
        exploration_exploitation_epsilon=exploration_exploitation_rate(
            train_loop_start_step
        ),
    )
    if train_loop_start_step != 0:
      model_state = restore_model_state(
          checkpoint_manager=checkpoint_manager,
          model_state=model_state,
          hyperparameters=hyperparameters,
      )
    pmapped_train_step, pmapped_evaluation_step = pmap_train_and_evaluation(
        hyperparameters=hyperparameters,
        agent=agent,
        reward=reward,
    )
    replicated_model_state = flax.jax_utils.replicate(model_state)
    return cls(
        train_rng_key=train_rng_key,
        learning_rate=learning_rate,
        exploration_exploitation_rate=exploration_exploitation_rate,
        tensorboard_writer=tensorboard_writer,
        train_loop_start_step=train_loop_start_step,
        checkpoint_manager=checkpoint_manager,
        pmapped_train_step=pmapped_train_step,
        pmapped_evaluation_step=pmapped_evaluation_step,
        model_state=model_state,
        replicated_model_state=replicated_model_state,
    )


def evaluate_on_step(
    *,
    hyperparameters: config_dict.ConfigDict,
    evaluation_data: tf.data.Dataset,
    training_blocks: TrainingBuildingBlocks,
    replicated_model_state: base_agent.BaseAgentState,
    step: int,
) -> None:
  """Evaluates an agent and saves the results to TensorBoard.

  Args:
    hyperparameters: Training hyperparameters.
    evaluation_data: An evaluation data pipeline.
    training_blocks: Building blocks of the training procedure.
    replicated_model_state: A replicated model state.
    step: The latest training step from the Orbax Checkpoint Manager.
  """
  evaluation_replicated_model_state = _sync_batch_statistics(
      model_state=replicated_model_state
  )
  evaluation_metrics_summary = evaluate(
      model_state=evaluation_replicated_model_state,
      evaluation_data=evaluation_data,
      pmapped_evaluation_step=training_blocks.pmapped_evaluation_step,
      evaluation_steps=hyperparameters.evaluation_steps,
  )
  for k, v in evaluation_metrics_summary.items():
    training_blocks.tensorboard_writer.scalar(k, v, step)


def update_model_state(
    *,
    batch: jnp.ndarray,
    training_blocks: TrainingBuildingBlocks,
    hyperparameters: config_dict.ConfigDict,
    replicated_model_state: base_agent.BaseAgentState,
    train_step_metrics: dict[str, float],
    step: int,
) -> tuple[base_agent.BaseAgentState, dict[str, float]]:
  """Returns an updated model state with train step metrics.

  Args:
    batch: An input train batch.
    training_blocks: Building blocks of the training procedure.
    hyperparameters: Training hyperparameters.
    replicated_model_state: A replicated model state.
    train_step_metrics: A mapping between train step metrics and their values.
    step: The latest training step from the Orbax Checkpoint Manager.
  """
  step_exploration_exploitation_rate = flax.jax_utils.replicate(
      train_step_metrics["step_exploration_exploitation_rate"]
  )
  step_clip_parameters_coefficient = flax.jax_utils.replicate(
      train_step_metrics["step_clip_parameters_coefficient"]
  )
  for epoch in range(hyperparameters.number_of_epochs + 1):
    permuted_trajectories = jax.random.permutation(
        jax.random.fold_in(training_blocks.train_rng_key, step + epoch),
        batch,
    )
    replicated_model_state, train_loss = training_blocks.pmapped_train_step(
        replicated_model_state,
        exploration_exploitation_rate=step_exploration_exploitation_rate,
        clip_parameters_coefficient=step_clip_parameters_coefficient,
        batch=common_utils.shard(permuted_trajectories),
    )
  train_step_metrics.update({"step_loss": train_loss.mean().item()})
  return replicated_model_state, train_step_metrics


def run_training_step(
    *,
    step: int,
    batch: jnp.ndarray,
    hyperparameters: config_dict.ConfigDict,
    replicated_model_state: base_agent.BaseAgentState,
    evaluation_data: tf.data.Dataset | None,
    training_blocks: TrainingBuildingBlocks,
) -> base_agent.BaseAgentState:
  """Returns an updated model state with train step metrics.

  Args:
    step: The latest training step from the Orbax Checkpoint Manager.
    batch: An input train batch.
    hyperparameters: Training hyperparameters.
    replicated_model_state: A replicated model state.
    evaluation_data: An evaluation data pipeline.
    training_blocks: Building blocks of the training procedure.
  """
  if evaluation_data and is_evaluation_step(
      step=step, hyperparameters=hyperparameters
  ):
    evaluate_on_step(
        hyperparameters=hyperparameters,
        evaluation_data=evaluation_data,
        training_blocks=training_blocks,
        replicated_model_state=replicated_model_state,
        step=step,
    )
  train_metrics_summary = calculate_train_step_metrics(
      step=step,
      learning_rate=training_blocks.learning_rate,
      exploration_exploitation_rate=training_blocks.exploration_exploitation_rate,
      hyperparameters=hyperparameters,
  )
  replicated_model_state, train_metrics_summary = update_model_state(
      batch=batch,
      training_blocks=training_blocks,
      hyperparameters=hyperparameters,
      replicated_model_state=replicated_model_state,
      train_step_metrics=train_metrics_summary,
      step=step,
  )
  for k, v in train_metrics_summary.items():
    training_blocks.tensorboard_writer.scalar(k, v, step)
  training_blocks.checkpoint_manager.save(
      step=step,
      items=_process_replicated_model_state(model_state=replicated_model_state),
  )
  training_blocks.checkpoint_manager.wait_until_finished()
  logging.info("Finished training step %d.", step)
  return replicated_model_state


def map_train_data(
    *,
    train_data: jnp.ndarray,
    hyperparameters: config_dict.ConfigDict,
) -> Mapping[str, jnp.ndarray]:
  """Returns mapped train arrays for the pretraining procedure.

  Args:
    train_data: An array with train data.
    hyperparameters: Training hyperparameters.
  """
  input_dimensions = hyperparameters.input_dimensions
  return dict(
      states=train_data[:, :input_dimensions],
      actions=train_data[:, input_dimensions : input_dimensions + 1],
      rewards=train_data[:, input_dimensions + 1 : input_dimensions + 2],
      values=train_data[:, input_dimensions + 2 : input_dimensions + 3],
      log_probabilities=train_data[
          :, input_dimensions + 3 : input_dimensions + 4
      ],
      dones=train_data[:, input_dimensions + 4 : input_dimensions + 5],
      attentive_transformer_losses=train_data[
          :, input_dimensions + 5 : input_dimensions + 6
      ],
  )


class BaseTrainer(metaclass=abc.ABCMeta):
  """Abstract parent class for all trainer classes.

  Attributes:
    agent: A training agent.
    reward: A training reward.
    hyperparameters: Training hyperparameters.
  """

  def __init__(
      self,
      *,
      agent: base_agent.BaseAgent,
      reward: base_reward.BaseReward,
      hyperparameters: config_dict.ConfigDict,
  ) -> None:
    """Initializes the Base Trainer class.

    Args:
      agent: A training agent.
      reward: A training reward.
      hyperparameters: Training hyperparameters.
    """
    self.agent = agent
    self.reward = reward
    self.hyperparameters = hyperparameters

  def train(
      self,
      *,
      train_data: Mapping[str, tf.Tensor | jnp.ndarray],
      evaluation_dataset: tf.data.Dataset | None = None,
      pretrain: bool = False,
      pretraining_blocks: TrainingBuildingBlocks | None = None,
      pretrain_model_state: base_agent.BaseAgentState | None = None,
      pretrain_step: int | None = None,
  ) -> base_agent.BaseAgentState:
    """Returns a model state after an end-to-end model training.

    Args:
      train_data: A batch with train data with collected experiences containing
        tensors / arrays of "states", "actions", "rewards", "values",
        "log_probabilities", "dones" and "attentive_transformer_losses". The
        dimensions of the arrays should be (batch_size, input_dimensions) for
        the "state" array and (batch_size, 1) for all the other arrays.
      evaluation_dataset: A dataset with tensors where the first N columns
        contain "state" and the last column contain the "target_action". The
        shape of the evaluation_dataset should be (evaluation_batch_size,
        input_dimensions + 1).
      pretrain: An indicator if the training occurs during pretraining session.
      pretraining_blocks: A dataclass with training blocks. It's only to be used
        during pretraining.
      pretrain_model_state: A model state to replicate. It's only to be used
        during pretraining.
      pretrain_step: A pretrain step. It's only to be used during pretraining.
    """
    if pretrain:
      if not pretraining_blocks:
        raise ValueError(
            "Pretraining requires pretraining blocks to be provided."
        )
      if not pretrain_model_state:
        raise ValueError(
            "Pretraining requires pretrain model state to be provided."
        )
      if not pretrain_step:
        raise ValueError("Pretraining requires pretrain step to be provided.")
      training_blocks = pretraining_blocks
      replicated_model_state = flax.jax_utils.replicate(pretrain_model_state)
      step = pretrain_step
    else:
      training_blocks = TrainingBuildingBlocks.prepare(
          hyperparameters=self.hyperparameters,
          agent=self.agent,
          reward=self.reward,
      )
      replicated_model_state = training_blocks.replicated_model_state
      step = training_blocks.train_loop_start_step + 1
    evaluation_data = initialize_evaluation_data(
        evaluation_dataset=evaluation_dataset,
        hyperparameters=self.hyperparameters,
    )
    trajectories = calculate_trajectories(
        batch=jax.tree_util.tree_map(jnp.asarray, train_data),
        hyperparameters=self.hyperparameters,
    )
    logging.info("Starting learning from the passed experience.")
    replicated_model_state = run_training_step(
        step=step,
        batch=trajectories,
        hyperparameters=self.hyperparameters,
        replicated_model_state=replicated_model_state,
        evaluation_data=evaluation_data,
        training_blocks=training_blocks,
    )
    if not pretrain:
      training_blocks.tensorboard_writer.close()
    logging.info("Finished learning.")
    return _process_replicated_model_state(model_state=replicated_model_state)

  def pretrain(
      self,
      *,
      pretrain_data: jnp.ndarray,
      evaluation_data: jnp.ndarray | None = None,
  ) -> base_agent.BaseAgentState:
    """Returns a model state after an end-to-end model training.

    Args:
      pretrain_data: A dataset with arrays where the first column is prediction
        seeds and the remaining columns of the array are train data. The shape
        of the pretrain_dataset should be (total_number_of_train_observations, 1
        + input_dimensions).
      evaluation_data: A dataset with arrays where the first N columns contain
        "state" and the last column contain the "target_action". The shape of
        the evaluation_data should be (total_number_of_evaluation_observations,
        input_dimensions + 1).
    """
    data_pipeline = data_pipelines.get_data_pipeline(
        data_pipeline_name=self.hyperparameters.data_pipeline_name
    )(hyperparameters=self.hyperparameters)
    train_pipeline_iterator = iter(
        data_pipeline.pretrain_data_pipeline(
            pretrain_data=pretrain_data,
        )
    )
    evaluation_pipeline = data_pipeline.evaluation_data_pipeline(
        evaluation_data=evaluation_data,
    )
    pretraining_blocks = TrainingBuildingBlocks.prepare(
        hyperparameters=self.hyperparameters,
        agent=self.agent,
        reward=self.reward,
    )
    model_state = pretraining_blocks.model_state
    train_loop_start_step = pretraining_blocks.train_loop_start_step + 1
    logging.info("Starting pretraining.")
    for step in tqdm.trange(
        train_loop_start_step, self.hyperparameters.train_steps + 1
    ):
      partial_pretrain_predict = functools.partial(
          self.agent.pretrain_predict,
          agent_state=model_state,
          calculate_pretrain_reward=self.reward.calculate_pretrain_reward,
      )
      batch = jnp.asarray(next(train_pipeline_iterator))
      train_data = jax.vmap(partial_pretrain_predict)(
          prediction_seed=batch[:, :1], batch=batch[:, 1:]
      )
      mapped_train_data = map_train_data(
          train_data=train_data, hyperparameters=self.hyperparameters
      )
      pretraining_blocks.tensorboard_writer.scalar(
          "training_reward",
          mapped_train_data["rewards"].sum().item(),
          step,
      )
      model_state = self.train(
          train_data=mapped_train_data,
          evaluation_dataset=evaluation_pipeline,
          pretrain=True,
          pretraining_blocks=pretraining_blocks,
          pretrain_model_state=model_state,
          pretrain_step=step,
      )
      logging.info("Finished pretraining step %d.", step)
    pretraining_blocks.tensorboard_writer.close()
    logging.info("Finished pretraining.")
    return model_state
