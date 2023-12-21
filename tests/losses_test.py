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

"""Tests for utility functions in losses.py."""
from absl.testing import absltest
from absl.testing import parameterized
import flax.linen as nn
import jax
import jax.numpy as jnp
from ml_collections.config_dict import config_dict
import optax
from optimus.agent_lib import base_agent
from optimus.trainer_lib import losses

_TEST_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        action_space_length=1,
        input_dimensions=5,
        value_function_coefficient=1,
        entropy_coefficient=1,
        lambda_sparse=1,
    )
)

_SINGLE_ACTION_BATCH = [[1, 2, 3, 4, 5, 10, 0.1, 1, 0.5, 0.2]]
_MULTI_ACTION_BATCH = [[1, 2, 3, 4, 5, 10, 11, 0.1, 1, 0.5, 0.2]]


class Model(nn.Module):

  @nn.compact
  def __call__(self, input_x, train=True):
    del train
    batch_normalization_output = nn.BatchNorm(
        use_running_average=False, momentum=0.9, epsilon=1e-5
    )(input_x)
    predicted_log_probabilities = nn.Dense(features=1, dtype=jnp.float32)(
        batch_normalization_output
    )
    values = jnp.asarray([[1]])
    attentive_transformer_losses = jnp.asarray(1)
    return predicted_log_probabilities, values, attentive_transformer_losses


class ProximalPolicyOptimizationLossTests(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="action_space_length_1",
          batch=_SINGLE_ACTION_BATCH,
          expected_outcome=[10],
          action_space_length=1,
      ),
      dict(
          testcase_name="action_space_length_2",
          batch=_MULTI_ACTION_BATCH,
          expected_outcome=[[[10]], [[11]]],
          action_space_length=2,
      ),
  )
  def test_process_batch(self, batch, action_space_length, expected_outcome):
    modified_hyperparameters = _TEST_HYPERPARAMETERS.to_dict()
    modified_hyperparameters.update(
        {"action_space_length": action_space_length}
    )
    actual_outcome = losses.process_batch(
        batch=jnp.asarray(batch),
        hyperparameters=config_dict.ConfigDict(modified_hyperparameters),
    )
    self.assertEqual(
        [i.tolist() for i in actual_outcome.actions], expected_outcome
    )

  def test_calculate_entropy_and_ratios_action_space_length_1(self):
    modified_hyperparameters = _TEST_HYPERPARAMETERS.to_dict()
    modified_hyperparameters.update({"action_space_length": 1})
    processed_batch = losses.process_batch(
        batch=jnp.asarray(_SINGLE_ACTION_BATCH),
        hyperparameters=config_dict.ConfigDict(modified_hyperparameters),
    )
    actual_outcome = losses.calculate_entropy_and_ratios(
        jnp.asarray([[0.9]]),
        actions=processed_batch.actions,
        original_log_probabilities=processed_batch.original_log_probabilities,
        action_space_length=1,
    )
    self.assertTrue(
        jnp.allclose(
            actual_outcome[0],
            jnp.asarray(-2.213647),
            rtol=1e-4,
            atol=1e-4,
        )
    )

  def test_calculate_entropy_and_ratios_action_space_length_2(self):
    modified_hyperparameters = _TEST_HYPERPARAMETERS.to_dict()
    modified_hyperparameters.update({"action_space_length": 2})
    processed_batch = losses.process_batch(
        batch=jnp.asarray(_MULTI_ACTION_BATCH),
        hyperparameters=config_dict.ConfigDict(modified_hyperparameters),
    )
    actual_outcome = losses.calculate_entropy_and_ratios(
        (jnp.asarray([[0.9]]), jnp.asarray([[0.1]])),
        actions=processed_batch.actions,
        original_log_probabilities=processed_batch.original_log_probabilities,
        action_space_length=2,
    )
    self.assertTrue(
        jnp.allclose(
            actual_outcome[0],
            jnp.asarray(-1.162082),
            rtol=1e-4,
            atol=1e-4,
        )
    )

  def test_calculate_tabnet_proximal_policy_optimization_loss(self):
    model = Model()
    parameters = model.init(
        jax.random.PRNGKey(0), jnp.array([1, 1, 1, 1, 1], dtype=jnp.float32)
    )
    model_state = base_agent.BaseAgentState.create(
        apply_fn=model.apply,
        params=parameters["params"],
        tx=optax.adam(0.1),
        batch_stats=parameters["batch_stats"],
        exploration_exploitation_epsilon=0.0,
    )
    actual_output, _ = (
        losses.calculate_tabnet_proximal_policy_optimization_loss(
            model_state.params,
            model_state.batch_stats,
            model_state.apply_fn,
            batch=jnp.asarray(_SINGLE_ACTION_BATCH),
            clip_parameters_coefficient=1,
            hyperparameters=_TEST_HYPERPARAMETERS,
        )
    )
    self.assertTrue(jnp.allclose(actual_output, jnp.asarray(-1.0)))


class LossesTests(absltest.TestCase):

  def test_get_loss(self):
    self.assertEqual(
        losses.get_loss(loss_name="tabnet_proximal_policy_optimization_loss"),
        losses.calculate_tabnet_proximal_policy_optimization_loss,
    )

  def test_get_loss_lookup_error(self):
    with self.assertRaisesRegex(LookupError, "Unrecognized loss name"):
      losses.get_loss(loss_name="other_loss")


if __name__ == "__main__":
  absltest.main()
