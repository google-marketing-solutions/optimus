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

"""Tests for utility functions in base_agent.py."""

import functools
import os

from absl.testing import absltest
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import numpy as np
import optax

from optimus.agent_lib import base_agent
from optimus.reward_lib import base_reward
from optimus.trainer_lib import base_trainer


_TEST_BASE_AGENT_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        input_dimensions=4,
        columns=("a", "b"),
        categorical_columns=("a",),
        categorical_dimensions=(2,),
        categorical_indexes=(1,),
        shuffle_size=1,
        batch_size=1,
        minibatch_denominator=1,
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
        reactions_dimensions=1,
    )
)


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

  def __init__(self, hyperparameters=_TEST_BASE_AGENT_HYPERPARAMETERS):
    super().__init__(hyperparameters=hyperparameters)

  def get_dummy_inputs(self):
    return np.zeros(self.hyperparameters.input_dimensions, dtype=np.float32)

  def build_flax_module(self):
    return MockModel()


class MockReward(base_reward.BaseReward):

  def __init__(self, hyperparameters=_TEST_BASE_AGENT_HYPERPARAMETERS):
    super().__init__(hyperparameters=hyperparameters)

  def calculate_reward(self, actions, reactions, sign_rewards):
    del sign_rewards
    return np.array(0)

  def calculate_pretrain_reward(self, batch, actions, sign_rewards):
    del batch, actions, sign_rewards
    return np.array([[0]])

  def calculate_evaluation_reward(self, predicted_actions, target_actions):
    return np.array(0)


class BaseAgentUtilityFunctionsTests(absltest.TestCase):

  def test_custom_train_state_with_batch_stats(self):
    model = MockModel()
    parameters = model.init(
        jax.random.PRNGKey(0), jnp.array([1, 2, 3, 4], dtype=jnp.float32)
    )
    model_state = base_agent.BaseAgentState.create(
        apply_fn=model.apply,
        params=parameters["params"],
        tx=optax.adam(0.1),
        batch_stats=parameters["batch_stats"],
        exploration_exploitation_epsilon=0.0,
    )
    self.assertEqual(
        model_state.params["BatchNorm_0"]["scale"].tolist(),
        [1.0, 1.0, 1.0, 1.0],
    )

  def test_apply_function(self):
    model = MockModel()
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
    actual_outcome = base_agent.apply_policy(
        parameters=model_state.params,
        apply_function=model_state.apply_fn,
        batch=jnp.array([[1, 2, 3, 4]], dtype=jnp.float32),
        batch_statistics=model_state.batch_stats,
        mutable=["batch_stats"],
    )
    self.assertEqual(actual_outcome[0][0].item(), 0.0)

  def test_predict_actions(self):
    actual_output = base_agent.predict_actions(
        exploration_exploitation_epsilon=0.0,
        prng_key=jax.random.PRNGKey(0),
        probabilities=jnp.asarray([0.1, 0.2, 0.7]),
    )
    self.assertEqual(actual_output.item(), 2)


class BaseAgentTests(absltest.TestCase):

  def test_build_flax_module(self):
    self.assertIsInstance(MockAgent().build_flax_module(), MockModel)

  def test_get_dummy_inputs(self):
    self.assertEqual(MockAgent().get_dummy_inputs().tolist(), [0, 0, 0, 0])

  def test_predict(self):
    model = MockModel()
    parameters = model.init(
        jax.random.PRNGKey(0), jnp.array([1, 2, 3, 4], dtype=jnp.float32)
    )
    model_state = base_agent.BaseAgentState.create(
        apply_fn=model.apply,
        params=parameters["params"],
        tx=optax.adam(0.1),
        batch_stats=parameters["batch_stats"],
        exploration_exploitation_epsilon=0.0,
    )
    agent = MockAgent()
    actual_output = agent.predict(
        agent_state=model_state,
        batch=jnp.asarray([0, 1, 2, 3]),
        prediction_seed=0,
    )
    self.assertEqual(actual_output.action.item(), 0)

  def test_evaluate_predict(self):
    model = MockModel()
    parameters = model.init(
        jax.random.PRNGKey(0), jnp.array([1, 2, 3, 4], dtype=jnp.float32)
    )
    model_state = base_agent.BaseAgentState.create(
        apply_fn=model.apply,
        params=parameters["params"],
        tx=optax.adam(0.1),
        batch_stats=parameters["batch_stats"],
        exploration_exploitation_epsilon=0.0,
    )
    agent = MockAgent()
    actual_output = agent.evaluate_predict(
        agent_state=model_state,
        batch=jnp.asarray([[0, 1, 2, 3]]),
    )
    self.assertEqual(actual_output.item(), 0)

  def test_pretrain_predict(self):
    agent = MockAgent()
    modified_hyperparameters_dictionary = (
        _TEST_BASE_AGENT_HYPERPARAMETERS.to_dict()
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
    model_state = base_trainer.initialize_model_state_for_prediction(
        agent=agent,
        hyperparameters=modified_hyperparameters,
    )
    partial_pretrain_predict = functools.partial(
        agent.pretrain_predict,
        agent_state=model_state,
        calculate_pretrain_reward=MockReward().calculate_pretrain_reward,
        prediction_seed=0,
    )
    actual_output = jax.vmap(partial_pretrain_predict)(
        batch=jnp.asarray([[1, 2, 3, 4]])
    )
    self.assertEqual(
        actual_output.tolist(),
        [[1.0, 2.0, 3.0, 4.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
    )

  def test_pretrain_predict_value_error(self):
    agent = MockAgent()
    modified_hyperparameters_dictionary = (
        _TEST_BASE_AGENT_HYPERPARAMETERS.to_dict()
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
    model_state = base_trainer.initialize_model_state_for_prediction(
        agent=agent,
        hyperparameters=modified_hyperparameters,
    )
    partial_pretrain_predict = functools.partial(
        agent.pretrain_predict,
        agent_state=model_state,
        calculate_pretrain_reward=MockReward().calculate_pretrain_reward,
        prediction_seed=0,
    )
    with self.assertRaisesRegex(ValueError, "batch must be one-dimensional"):
      jax.vmap(partial_pretrain_predict)(batch=jnp.asarray([[[1, 2, 3, 4]]]))


if __name__ == "__main__":
  absltest.main()
