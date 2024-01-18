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

"""Tests for utility functions in base_trainer.py."""

import functools
import os

from absl.testing import absltest
from absl.testing import parameterized
import flax
import flax.linen as nn
from flax.metrics import tensorboard
from flax.training import common_utils
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
import optax
import tensorflow as tf

from optimus.agent_lib import base_agent
from optimus.data_pipeline_lib import base_data_pipeline
from optimus.reward_lib import base_reward
from optimus.trainer_lib import base_trainer

_TEST_BASE_TRAINER_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        input_dimensions=4,
        columns=("a", "b"),
        categorical_columns=("a",),
        categorical_dimensions=(2,),
        categorical_indexes=(1,),
        shuffle_size=1,
        batch_size=1,
        evaluation_batch_size=1,
        sign_rewards=True,
        action_space=1,
        action_dir_path=None,
        skip_action_dir=None,
        attention_gamma=1.3,
        prediction_layer_dimension=8,
        attention_layer_dimension=8,
        successive_network_steps=3,
        categorical_embedding_dimensions=1,
        independent_glu_layers=2,
        shared_glu_layers=2,
        epsilon=1e-15,
        virtual_batch_size=128,
        momentum=0.98,
        mask_type="sparsemax",
        shared_decoder_layers=1,
        independent_decoder_layers=1,
        model_data_type="float32",
        data_pipeline_name="base_data_pipeline",
        actions_name="base_actions",
        agent_name="tabnet",
        reward_name="base_reward",
        trainer_name="base_trainer",
        train_steps=1,
        evaluation_frequency=1,
        evaluation_steps=1,
        gae_gamma=0.99,
        gae_lambda=0.95,
        clip_parameters=True,
        clip_parameters_coefficient=0.1,
        value_function_coefficient=0.5,
        entropy_coefficient=0.02,
        lambda_sparse=1e-3,
        checkpoint_frequency=500,
        checkpoints_kept=10,
        training_rng_seed=-1,
        loss="tabnet_proximal_policy_optimization_loss",
        learning_rate_hyperparameters={
            "initial_value": 0.01,
            "schedule": "linear_schedule",
            "end_value": 0.0,
            "transition_steps": 1,
        },
        exploration_exploitation_hyperparameters={
            "schedule": "constant_schedule",
            "initial_value": 0.0,
        },
        optimizer="adam",
        optimizer_hyperparameters={
            "b1": 0.9,
            "b2": 0.999,
            "eps": 1e-8,
            "eps_root": 0.0,
        },
        train_dataset_size=1,
        number_of_epochs=1,
        action_space_length=1,
        replace_nans_in_prediction=True,
    )
)

_TRAIN_DATA = np.asarray([[0.0, 1.0, 2.0, 3.0, 4.0]])
_EVALUATION_DATA = np.asarray([[1.0, 2.0, 3.0, 4.0, 1.0]])


class MockState(base_data_pipeline.BaseDataPipeline):

  def __init__(
      self,
      seed=0,
      hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS,
      train_dataset=_TRAIN_DATA,
      evaluation_dataset=_EVALUATION_DATA,
  ):
    super().__init__(
        hyperparameters=hyperparameters,
    )

  def load_train_and_evaluation_pipelines(self):
    return tf.data.Dataset.from_tensor_slices({
        "states": [[[0.0, 1.0, 2.0, 3.0]]],
        "actions": [[[0]]],
        "rewards": [[[0.0]]],
        "values": [[[0.0]]],
        "log_probabilities": [[[0.0]]],
        "dones": [[[0.0]]],
        "attentive_transformer_losses": [[[0.0]]],
    }), tf.data.Dataset.from_tensor_slices(
        {"states": [[[0, 1, 2, 3]]], "target_actions": [[[0]]]}
    )


class MockReward(base_reward.BaseReward):

  def __init__(self, hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS):
    super().__init__(hyperparameters=hyperparameters)

  def calculate_reward(self, actions, reactions, sign_rewards):
    del sign_rewards
    return np.array(0)

  def calculate_pretrain_reward(self, batch, actions, sign_rewards):
    del batch, actions
    return np.array([[0]])

  def calculate_evaluation_reward(self, predicted_actions, target_actions):
    return np.array(0)


class MockModel(nn.Module):

  @nn.compact
  def __call__(self, input_x, **kwargs):
    del kwargs
    batch_normalization_output = nn.BatchNorm(
        use_running_average=True, momentum=0.9, epsilon=1e-5, dtype=jnp.float32
    )(input_x)
    _ = nn.Dense(1)(batch_normalization_output)
    log_probabilities = jnp.asarray([[0.0]])
    values = jnp.asarray([[0.0]])
    attentive_transformer_loss = jnp.asarray([[0.0]])
    return (log_probabilities, values, attentive_transformer_loss)


class MockAgent(base_agent.BaseAgent):

  def __init__(self, hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS):
    super().__init__(hyperparameters=hyperparameters)

  def get_dummy_inputs(self):
    return np.zeros(self.hyperparameters.input_dimensions, dtype=np.float32)

  def build_flax_module(self):
    return MockModel()


class BaseTrainerUtilityFunctionsTests(parameterized.TestCase):

  def test_calculate_advantages(self):
    rewards = jnp.asarray([1, 2, 3, 4])
    terminal_masks = jnp.asarray([1, 0, 1, 0])
    values = jnp.asarray([0.1, 0.2, 0.3, 0.4, 0.4])
    combined_arrays = jnp.stack(
        (rewards, terminal_masks, values[:-1], values[1:]), axis=1
    )
    generalized_advantage_estimation = 0.0
    partial_calculate_advantages = functools.partial(
        base_trainer.calculate_advantages,
        gae_gamma=_TEST_BASE_TRAINER_HYPERPARAMETERS.gae_gamma,
        gae_lambda=_TEST_BASE_TRAINER_HYPERPARAMETERS.gae_lambda,
    )
    _, actual_output = jax.lax.scan(
        partial_calculate_advantages,
        generalized_advantage_estimation,
        combined_arrays,
    )
    self.assertTrue(
        jnp.array_equal(
            actual_output,
            jnp.array([1.098, 1.8, 4.7889, 3.6], dtype=jnp.float32),
        )
    )

  def test_calculate_trajectories(self):
    experience_batch = dict(
        states=jnp.array([[1, 2]], dtype=jnp.float32),
        actions=jnp.array([[2]], dtype=jnp.float32),
        rewards=jnp.array([[3]], dtype=jnp.float32),
        values=jnp.array([[4]], dtype=jnp.float32),
        log_probabilities=jnp.array([[5]], dtype=jnp.float32),
        dones=jnp.array([[0]], dtype=jnp.float32),
        attentive_transformer_losses=jnp.array([[1]], dtype=jnp.float32),
    )
    actual_output = base_trainer.calculate_trajectories(
        batch=experience_batch,
        hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS,
    )
    expected_output = jnp.array(
        [[1.0, 2.0, 2.0, 5.0, 3.0, -1.0, 1.0]], dtype=jnp.float32
    )
    self.assertTrue(jnp.array_equal(actual_output, expected_output))

  def test_calculate_train_step_metrics(self):
    actual_output = base_trainer.calculate_train_step_metrics(
        step=1,
        learning_rate=optax.constant_schedule(0.1),
        exploration_exploitation_rate=optax.constant_schedule(0.2),
        hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS,
    )
    self.assertEqual(actual_output["step_learning_rate"], 0.1)

  def test_train_step(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    model = MockModel()
    parameters = model.init(
        jax.random.PRNGKey(0), jnp.array([[1, 2, 3, 4]], dtype=jnp.float32)
    )
    train_state = base_trainer.initialize_model_state(
        model=model,
        initial_parameters=parameters,
        optimizer=optax.adam(0.1),
        exploration_exploitation_epsilon=0.0,
    )

    def mock_loss_function(*args, **kwargs):
      del args
      del kwargs
      batch_stats = {
          "batch_stats": flax.core.freeze({
              "mean": jnp.array([[0.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
              "var": jnp.array([[1.0, 1.0, 1.0, 1.0]], dtype=jnp.float32),
          })
      }
      return 0.1, batch_stats

    pmapped_train_step = jax.pmap(
        functools.partial(
            base_trainer.train_step,
            hyperparameters=modified_hyperparameters,
            loss_function=mock_loss_function,
        ),
        axis_name="batch",
    )
    replicated_model_state = flax.jax_utils.replicate(train_state)
    trajectories = base_trainer.calculate_trajectories(
        batch={
            "states": jnp.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=jnp.float32),
            "actions": jnp.asarray([[0]], dtype=jnp.int32),
            "rewards": jnp.asarray([[0.0]], dtype=jnp.float32),
            "values": jnp.asarray([[0.0]], dtype=jnp.float32),
            "log_probabilities": jnp.asarray([[0.0]], dtype=jnp.float32),
            "dones": jnp.asarray([[0.0]], dtype=jnp.float32),
            "attentive_transformer_losses": jnp.asarray(
                [[0.0]], dtype=jnp.float32
            ),
        },
        hyperparameters=modified_hyperparameters,
    )
    actual_output = pmapped_train_step(
        replicated_model_state,
        exploration_exploitation_rate=flax.jax_utils.replicate(0.0),
        clip_parameters_coefficient=flax.jax_utils.replicate(0.0),
        batch=common_utils.shard(trajectories),
    )
    self.assertEqual(actual_output[1], 0.1)

  def test_evaluate(self):
    model = nn.BatchNorm(
        use_running_average=False, momentum=0.9, epsilon=1e-5, dtype=jnp.float32
    )
    parameters = model.init(
        jax.random.PRNGKey(0), jnp.array([[1, 2, 3, 4]], dtype=jnp.float32)
    )
    model_state = base_agent.BaseAgentState.create(
        apply_fn=model.apply,
        params=parameters["params"],
        tx=optax.adam(0.1),
        batch_stats=parameters["batch_stats"],
        exploration_exploitation_epsilon=0.0,
    )
    replicated_model_state = flax.jax_utils.replicate(model_state)

    def mock_evaluation_function(agent_state, batch):
      del agent_state, batch
      return 1

    pmapped_evaluation_step = jax.pmap(
        mock_evaluation_function, axis_name="batch", donate_argnums=(0,)
    )
    evaluation_data = tf.data.Dataset.from_tensor_slices({
        "states": tf.constant([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=jnp.float32),
        "target_actions": tf.constant([[1], [2]], dtype=jnp.float32),
    }).batch(2)
    actual_output = base_trainer.evaluate(
        model_state=replicated_model_state,
        evaluation_data=evaluation_data,
        pmapped_evaluation_step=pmapped_evaluation_step,
        evaluation_steps=1,
    )
    self.assertTrue(actual_output["evaluation_reward"].item(), 1)

  def test_initialize_evaluation_data(self):
    evaluation_dataset = tf.data.Dataset.from_tensor_slices(
        {"states": [[[0, 1, 2, 3]]], "target_actions": [[[0]]]}
    )
    evaluation_data = base_trainer.initialize_evaluation_data(
        evaluation_dataset=evaluation_dataset,
        hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS,
    )
    actual_output = (
        list(evaluation_data.take(1))[0]["target_actions"].numpy().item()
    )
    self.assertEqual(actual_output, 0)

  @parameterized.named_parameters(
      dict(
          testcase_name="evaluation_frequency",
          key_to_remove="evaluation_frequency",
      ),
      dict(
          testcase_name="evaluation_steps",
          key_to_remove="evaluation_steps",
      ),
  )
  def test_initialize_evaluation_data_value_error(self, key_to_remove):
    hyperparameters_dictionary = _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    del hyperparameters_dictionary[key_to_remove]
    modified_hyperparameters = config_dict.ConfigDict(
        hyperparameters_dictionary
    )
    evaluation_dataset = tf.data.Dataset.from_tensor_slices(
        {"states": [[[0, 1, 2, 3]]], "target_actions": [[[0]]]}
    )
    with self.assertRaisesRegex(
        ValueError, "evaluation_frequency and evaluation_steps hyperparameters"
    ):
      base_trainer.initialize_evaluation_data(
          evaluation_dataset=evaluation_dataset,
          hyperparameters=modified_hyperparameters,
      )

  def test_initialize_model(self):
    _, parameters = base_trainer.initialize_model(
        initialization_rng_key=jax.random.PRNGKey(0),
        agent=MockAgent(),
    )
    actual_output = parameters["params"]["Dense_0"]["bias"].item()
    self.assertEqual(actual_output, 0.0)

  def test_initialize_schedule(self):
    actual_output = base_trainer.initialize_schedule(
        hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS.learning_rate_hyperparameters
    )
    self.assertEqual(actual_output(0), 0.01)

  def test_initialize_tensorboard_writer(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    tensorboard_writer = base_trainer.initialize_tensorboard_writer(
        hyperparameters=modified_hyperparameters
    )
    self.assertIsInstance(tensorboard_writer, tensorboard.SummaryWriter)

  def test_initialize_checkpointing(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    checkpoint_manager, latest_training_step = (
        base_trainer.initialize_checkpointing(
            hyperparameters=modified_hyperparameters
        )
    )
    latest_training_step_checkpointer = checkpoint_manager.latest_step()
    actual_output = [latest_training_step, latest_training_step_checkpointer]
    self.assertEqual(actual_output, [0, None])

  def test_restore_model_state(self):
    model = nn.BatchNorm(
        use_running_average=False, momentum=0.9, epsilon=1e-5, dtype=jnp.float32
    )
    parameters = model.init(
        jax.random.PRNGKey(0), jnp.array([[1, 2, 3, 4]], dtype=jnp.float32)
    )
    train_state = base_trainer.initialize_model_state(
        model=model,
        initial_parameters=parameters,
        optimizer=optax.adam(0.1),
        exploration_exploitation_epsilon=0.0,
    )
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    checkpoint_manager, _ = base_trainer.initialize_checkpointing(
        hyperparameters=modified_hyperparameters
    )
    checkpoint_manager.save(
        step=1,
        items=train_state,
        force=True,
    )
    checkpoint_manager.wait_until_finished()
    actual_output = (
        base_trainer.restore_model_state(
            checkpoint_manager=checkpoint_manager,
            model_state=train_state,
            hyperparameters=modified_hyperparameters,
        )
        .params["scale"]
        .tolist()
    )
    self.assertEqual(actual_output, [1.0, 1.0, 1.0, 1.0])

  def test_pmap_train_and_evaluation(self):
    pmapped_train_step, pmapped_evaluation_step = (
        base_trainer.pmap_train_and_evaluation(
            agent=MockAgent(),
            reward=MockReward(),
            hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS,
        )
    )
    actual_output = [
        type(pmapped_train_step).__name__,
        type(pmapped_evaluation_step).__name__,
    ]
    self.assertEqual(actual_output, ["PmapFunction", "PmapFunction"])

  @parameterized.named_parameters(
      dict(
          testcase_name="final_step",
          step=0,
      ),
      dict(
          testcase_name="evaluation_metrics_summary_step",
          step=1,
      ),
  )
  def test_is_evaluation_step(self, step):
    self.assertEqual(
        base_trainer.is_evaluation_step(
            step=step, hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS
        ),
        True,
    )

  def test_initialize_model_state(self):
    model = MockModel()
    parameters = model.init(
        jax.random.PRNGKey(0), jnp.array([[1, 2, 3, 4]], dtype=jnp.float32)
    )
    model_state = base_trainer.initialize_model_state(
        model=model,
        initial_parameters=parameters,
        optimizer=optax.adam(0.1),
        exploration_exploitation_epsilon=0.0,
    )
    self.assertEqual(
        model_state.params["BatchNorm_0"]["scale"].tolist(),
        [1.0, 1.0, 1.0, 1.0],
    )

  def test_initialize_model_state_for_prediction(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    actual_output = base_trainer.initialize_model_state_for_prediction(
        agent=MockAgent(),
        hyperparameters=modified_hyperparameters,
    )
    self.assertEqual(actual_output.step, 0)

  def test_prepare_training_building_blocks(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    actual_output = base_trainer.TrainingBuildingBlocks.prepare(
        agent=MockAgent(),
        reward=MockReward(),
        hyperparameters=modified_hyperparameters,
    )
    self.assertEqual(actual_output.learning_rate(0), 0.01)

  def test_evaluate_on_step(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    training_blocks = base_trainer.TrainingBuildingBlocks.prepare(
        agent=MockAgent(),
        reward=MockReward(),
        hyperparameters=modified_hyperparameters,
    )
    base_trainer.evaluate_on_step(
        step=0,
        hyperparameters=modified_hyperparameters,
        evaluation_data=tf.data.Dataset.from_tensor_slices(
            {"states": [[[0, 1, 2, 3]]], "target_actions": [[[0]]]}
        ),
        training_blocks=training_blocks,
        replicated_model_state=training_blocks.replicated_model_state,
    )
    self.assertTrue(
        any(
            item.startswith("events")
            for item in tf.io.gfile.listdir(
                os.path.join(directory, "artifacts")
            )
        )
    )

  def test_update_model_state(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    training_blocks = base_trainer.TrainingBuildingBlocks.prepare(
        agent=MockAgent(),
        reward=MockReward(),
        hyperparameters=modified_hyperparameters,
    )
    train_metrics = dict(
        step_clip_parameters_coefficient=0.0,
        step_exploration_exploitation_rate=0.0,
        step_learning_rate=0.0,
    )
    trajectories = base_trainer.calculate_trajectories(
        batch={
            "states": jnp.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=jnp.float32),
            "actions": jnp.asarray([[0]], dtype=jnp.int32),
            "rewards": jnp.asarray([[0.0]], dtype=jnp.float32),
            "values": jnp.asarray([[0.0]], dtype=jnp.float32),
            "log_probabilities": jnp.asarray([[0.0]], dtype=jnp.float32),
            "dones": jnp.asarray([[0.0]], dtype=jnp.float32),
            "attentive_transformer_losses": jnp.asarray(
                [[0.0]], dtype=jnp.float32
            ),
        },
        hyperparameters=modified_hyperparameters,
    )
    actual_output = base_trainer.update_model_state(
        batch=trajectories,
        training_blocks=training_blocks,
        hyperparameters=modified_hyperparameters,
        replicated_model_state=training_blocks.replicated_model_state,
        train_step_metrics=train_metrics,
        step=1,
    )
    self.assertEqual(actual_output[1]["step_loss"], -0.0)

  def test_run_training_step(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    training_blocks = base_trainer.TrainingBuildingBlocks.prepare(
        agent=MockAgent(),
        reward=MockReward(),
        hyperparameters=modified_hyperparameters,
    )
    trajectories = base_trainer.calculate_trajectories(
        batch={
            "states": jnp.asarray([[0.0, 1.0, 2.0, 3.0]], dtype=jnp.float32),
            "actions": jnp.asarray([[0]], dtype=jnp.int32),
            "rewards": jnp.asarray([[0.0]], dtype=jnp.float32),
            "values": jnp.asarray([[0.0]], dtype=jnp.float32),
            "log_probabilities": jnp.asarray([[0.0]], dtype=jnp.float32),
            "dones": jnp.asarray([[0.0]], dtype=jnp.float32),
            "attentive_transformer_losses": jnp.asarray(
                [[0.0]], dtype=jnp.float32
            ),
        },
        hyperparameters=modified_hyperparameters,
    )
    actual_output = base_trainer.run_training_step(
        step=0,
        batch=trajectories,
        hyperparameters=modified_hyperparameters,
        replicated_model_state=training_blocks.replicated_model_state,
        evaluation_data=tf.data.Dataset.from_tensor_slices(
            {"states": [[[0, 1, 2, 3]]], "target_actions": [[[0]]]}
        ),
        training_blocks=training_blocks,
    )
    self.assertEqual(
        actual_output.params["BatchNorm_0"]["bias"].tolist(),
        [[0.0, 0.0, 0.0, 0.0]],
    )

  def test_map_train_data(self):
    actual_output = base_trainer.map_train_data(
        train_data=jnp.asarray(
            [[1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
        ),
        hyperparameters=_TEST_BASE_TRAINER_HYPERPARAMETERS,
    )
    self.assertEqual(actual_output["states"].tolist(), [[1.0, 2.0, 3.0, 4.0]])


class BaseTrainerTests(absltest.TestCase):

  def test_train(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    trainer = base_trainer.BaseTrainer(
        agent=MockAgent(),
        reward=MockReward(),
        hyperparameters=modified_hyperparameters,
    )
    train_dataset = tf.data.Dataset.from_tensor_slices({
        "states": [[[0.0, 1.0, 2.0, 3.0]]],
        "actions": [[[0]]],
        "rewards": [[[0.0]]],
        "values": [[[0.0]]],
        "log_probabilities": [[[0.0]]],
        "dones": [[[0.0]]],
        "attentive_transformer_losses": [[[0.0]]],
    })
    evaluation_dataset = tf.data.Dataset.from_tensor_slices(
        {"states": [[[0, 1, 2, 3]]], "target_actions": [[[0]]]}
    )
    for train_batch in iter(train_dataset):
      actual_output = trainer.train(
          train_data=train_batch, evaluation_dataset=evaluation_dataset
      )
    self.assertEqual(
        actual_output.params["BatchNorm_0"]["bias"].tolist(),
        [0.0, 0.0, 0.0, 0.0],
    )

  def test_train_pretraining_blocks_value_error(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    trainer = base_trainer.BaseTrainer(
        agent=MockAgent(),
        reward=MockReward(),
        hyperparameters=modified_hyperparameters,
    )
    train_dataset = tf.data.Dataset.from_tensor_slices({
        "states": [[[0.0, 1.0, 2.0, 3.0]]],
        "actions": [[[0]]],
        "rewards": [[[0.0]]],
        "values": [[[0.0]]],
        "log_probabilities": [[[0.0]]],
        "dones": [[[0.0]]],
        "attentive_transformer_losses": [[[0.0]]],
    })
    evaluation_dataset = tf.data.Dataset.from_tensor_slices(
        {"states": [[[0, 1, 2, 3]]], "target_actions": [[[0]]]}
    )
    pretraining_blocks = base_trainer.TrainingBuildingBlocks.prepare(
        hyperparameters=modified_hyperparameters,
        agent=MockAgent(),
        reward=MockReward(),
    )
    with self.assertRaisesRegex(ValueError, "requires pretraining blocks"):
      for train_batch in iter(train_dataset):
        trainer.train(
            train_data=train_batch,
            evaluation_dataset=evaluation_dataset,
            pretrain=True,
            pretraining_blocks=None,
            pretrain_model_state=pretraining_blocks.model_state,
            pretrain_step=1,
        )

  def test_train_pretrain_model_state_value_error(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    trainer = base_trainer.BaseTrainer(
        agent=MockAgent(),
        reward=MockReward(),
        hyperparameters=modified_hyperparameters,
    )
    train_dataset = tf.data.Dataset.from_tensor_slices({
        "states": [[[0.0, 1.0, 2.0, 3.0]]],
        "actions": [[[0]]],
        "rewards": [[[0.0]]],
        "values": [[[0.0]]],
        "log_probabilities": [[[0.0]]],
        "dones": [[[0.0]]],
        "attentive_transformer_losses": [[[0.0]]],
    })
    evaluation_dataset = tf.data.Dataset.from_tensor_slices(
        {"states": [[[0, 1, 2, 3]]], "target_actions": [[[0]]]}
    )
    pretraining_blocks = base_trainer.TrainingBuildingBlocks.prepare(
        hyperparameters=modified_hyperparameters,
        agent=MockAgent(),
        reward=MockReward(),
    )
    with self.assertRaisesRegex(ValueError, "requires pretrain model state"):
      for train_batch in iter(train_dataset):
        trainer.train(
            train_data=train_batch,
            evaluation_dataset=evaluation_dataset,
            pretrain=True,
            pretraining_blocks=pretraining_blocks,
            pretrain_model_state=None,
            pretrain_step=1,
        )

  def test_train_pretrain_step_value_error(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    trainer = base_trainer.BaseTrainer(
        agent=MockAgent(),
        reward=MockReward(),
        hyperparameters=modified_hyperparameters,
    )
    train_dataset = tf.data.Dataset.from_tensor_slices({
        "states": [[[0.0, 1.0, 2.0, 3.0]]],
        "actions": [[[0]]],
        "rewards": [[[0.0]]],
        "values": [[[0.0]]],
        "log_probabilities": [[[0.0]]],
        "dones": [[[0.0]]],
        "attentive_transformer_losses": [[[0.0]]],
    })
    evaluation_dataset = tf.data.Dataset.from_tensor_slices(
        {"states": [[[0, 1, 2, 3]]], "target_actions": [[[0]]]}
    )
    pretraining_blocks = base_trainer.TrainingBuildingBlocks.prepare(
        hyperparameters=modified_hyperparameters,
        agent=MockAgent(),
        reward=MockReward(),
    )
    with self.assertRaisesRegex(ValueError, "requires pretrain step"):
      for train_batch in iter(train_dataset):
        trainer.train(
            train_data=train_batch,
            evaluation_dataset=evaluation_dataset,
            pretrain=True,
            pretraining_blocks=pretraining_blocks,
            pretrain_model_state=pretraining_blocks.model_state,
            pretrain_step=None,
        )

  def test_pretrain(self):
    modified_hyperparameters_dictionary = (
        _TEST_BASE_TRAINER_HYPERPARAMETERS.to_dict()
    )
    directory = self.create_tempdir().full_path
    modified_paths = dict(
        checkpoint_directory=os.path.join(directory, "checkpoints"),
        artifact_directory=os.path.join(directory, "artifacts"),
    )
    modified_hyperparameters_dictionary.update(modified_paths)
    modified_hyperparameters = config_dict.ConfigDict(
        modified_hyperparameters_dictionary
    )
    trainer = base_trainer.BaseTrainer(
        agent=MockAgent(),
        reward=MockReward(),
        hyperparameters=modified_hyperparameters,
    )
    actual_result = trainer.pretrain(
        pretrain_data=_TRAIN_DATA, evaluation_data=_EVALUATION_DATA
    )
    self.assertEqual(actual_result.step, 2)


if __name__ == "__main__":
  absltest.main()
