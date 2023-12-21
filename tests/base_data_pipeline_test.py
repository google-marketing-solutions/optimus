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

"""Tests for utility functions in base_data_pipeline_class.py."""
from absl.testing import absltest
from absl.testing import parameterized
from ml_collections.config_dict import config_dict
import numpy as np
import tensorflow as tf

from optimus.data_pipeline_lib import base_data_pipeline

_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS = config_dict.ConfigDict(
    dict(
        input_dimensions=2,
        columns=("a", "b"),
        categorical_columns=("a",),
        categorical_dimensions={"a": 2},
        categorical_indexes=(1),
        shuffle_size=1,
        batch_size=1,
        evaluation_batch_size=1,
        train_dataset_size=2,
        train_steps=1,
        sign_rewards=True,
        training_rng_seed=0,
    )
)
_TRAIN_DATA = np.asarray([[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
_PRETRAIN_DATA = np.asarray([[0.0, 1.0]])
_EVALUATION_DATA = np.asarray([[0.0, 1.0, 0.0]])


def _mock_calculate_reward(
    predicted_actions: tf.Tensor,
    reactions: tf.Tensor,
    sign_rewards: bool = True,
) -> tf.Tensor:
  del predicted_actions, reactions, sign_rewards
  return tf.constant([0.0], dtype=tf.float32)


class BaseDataPipelineUtilityTests(parameterized.TestCase):

  @parameterized.named_parameters(
      dict(
          testcase_name="train",
          split="train",
          data=[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
      ),
      dict(
          testcase_name="evaluation",
          split="evaluation",
          data=[0.0, 1.0, 0.0],
      ),
  )
  def test_process_data(self, split, data):
    actual_output = (
        base_data_pipeline.process_data(
            tensor=tf.constant(data, dtype=tf.float32),
            split=split,
            hyperparameters=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS,
            reward_calculation_function=_mock_calculate_reward,
        )["states"]
        .numpy()
        .tolist()
    )
    self.assertEqual(actual_output, [0.0, 1.0])

  def test_process_data_value_error(self):
    with self.assertRaisesRegex(ValueError, "Split must be either 'train'"):
      base_data_pipeline.process_data(
          tensor=tf.constant([1, 2, 3], dtype=tf.float32),
          split="other_split",
          hyperparameters=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS,
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="batch_size_processes",
          process_count=3,
          number_of_devices=1,
          batch_size=10,
          evaluation_batch_size=1,
          train_dataset_size=1,
          train_steps=1,
          error="must divide batch_size",
      ),
      dict(
          testcase_name="train_dataset_size_batch_size_train_steps",
          process_count=1,
          number_of_devices=1,
          batch_size=5,
          evaluation_batch_size=1,
          train_dataset_size=10,
          train_steps=100,
          error="train_dataset_size=",
      ),
      dict(
          testcase_name="evaluation_batch_size_processes",
          process_count=3,
          number_of_devices=1,
          batch_size=3,
          evaluation_batch_size=10,
          train_dataset_size=3,
          train_steps=1,
          error="must divide evaluation_batch_size",
      ),
      dict(
          testcase_name="per_host_batch_size_devices",
          process_count=2,
          number_of_devices=3,
          batch_size=4,
          evaluation_batch_size=2,
          train_dataset_size=4,
          train_steps=1,
          error="must divide per_host_batch_size",
      ),
      dict(
          testcase_name="per_host_evaluation_batch_size_devices",
          process_count=2,
          number_of_devices=2,
          batch_size=4,
          evaluation_batch_size=6,
          train_dataset_size=4,
          train_steps=1,
          error="must divide per_host_evaluation_batch_size",
      ),
  )
  def test_calculate_train_and_evaluation_batch_size_value_error(
      self,
      process_count,
      number_of_devices,
      batch_size,
      evaluation_batch_size,
      train_dataset_size,
      train_steps,
      error,
  ):
    modified_hyperparameters = (
        _TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS.to_dict()
    )
    parameters_to_change = dict(
        batch_size=batch_size,
        evaluation_batch_size=evaluation_batch_size,
        train_dataset_size=train_dataset_size,
        train_steps=train_steps,
    )
    modified_hyperparameters.update(parameters_to_change)
    with self.assertRaisesRegex(ValueError, error):
      base_data_pipeline.calculate_train_and_evaluation_batch_size(
          process_count=process_count,
          number_of_devices=number_of_devices,
          hyperparameters=config_dict.ConfigDict(modified_hyperparameters),
      )

  def test_calculate_train_and_evaluation_batch_size(self):
    actual_output = (
        base_data_pipeline.calculate_train_and_evaluation_batch_size(
            process_count=1,
            number_of_devices=1,
            hyperparameters=config_dict.ConfigDict(
                _TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS
            ),
        )
    )
    self.assertEqual(actual_output, (1, 1))


class BaseDataPipelineTests(parameterized.TestCase):

  def test_base_data_pipeline(self):
    base_data_pipeline_class = base_data_pipeline.BaseDataPipeline(
        hyperparameters=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS,
    )
    self.assertIsInstance(
        base_data_pipeline_class, base_data_pipeline.BaseDataPipeline
    )

  def test_build_tensorflow_pipeline_value_error(self):
    base_data_pipeline_class = base_data_pipeline.BaseDataPipeline(
        hyperparameters=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS,
    )
    with self.assertRaisesRegex(ValueError, "Split must be either 'train'"):
      base_data_pipeline_class.build_tensorflow_pipeline(
          data=_TRAIN_DATA, split="other_split"
      )

  @parameterized.named_parameters(
      dict(
          testcase_name="train",
          data=_TRAIN_DATA,
          split="train",
          reward_calculation_function_input=_mock_calculate_reward,
      ),
      dict(
          testcase_name="evaluation",
          data=_EVALUATION_DATA,
          split="evaluation",
          reward_calculation_function_input=None,
      ),
  )
  def test_build_tensorflow_pipeline(
      self, split, data, reward_calculation_function_input
  ):
    base_data_pipeline_class = base_data_pipeline.BaseDataPipeline(
        hyperparameters=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS,
    )
    actual_output = base_data_pipeline_class.build_tensorflow_pipeline(
        data=data,
        batch_size=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS.batch_size,
        split=split,
        reward_calculation_function=reward_calculation_function_input,
    )
    self.assertEqual(
        list(actual_output.take(1))[0]["states"].numpy().tolist(), [[0.0, 1.0]]
    )

  def test_per_host_batch_sizes(self):
    base_data_pipeline_class = base_data_pipeline.BaseDataPipeline(
        hyperparameters=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS,
    )
    self.assertEqual(base_data_pipeline_class.per_host_batch_sizes, (1, 1))

  def test_train_data_pipeline(self):
    base_data_pipeline_class = base_data_pipeline.BaseDataPipeline(
        hyperparameters=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS,
    )
    train_dataset = base_data_pipeline_class.train_data_pipeline(
        train_data=_TRAIN_DATA,
        reward_calculation_function=_mock_calculate_reward,
    )
    actual_output = [
        list(train_dataset.take(1))[0]["states"].numpy().tolist(),
    ]
    self.assertEqual(actual_output, [[[0.0, 1.0]]])

  def test_pretrain_data_pipeline(self):
    base_data_pipeline_class = base_data_pipeline.BaseDataPipeline(
        hyperparameters=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS,
    )
    pretrain_dataset = base_data_pipeline_class.pretrain_data_pipeline(
        pretrain_data=_PRETRAIN_DATA,
    )
    actual_output = [
        list(pretrain_dataset.take(1))[0].numpy().tolist(),
    ]
    self.assertEqual(actual_output, [[[0.0, 1.0]]])

  def test_evaluation_data_pipeline(self):
    base_data_pipeline_class = base_data_pipeline.BaseDataPipeline(
        hyperparameters=_TEST_BASE_DATA_PIPELINE_HYPERPARAMETERS,
    )
    evaluation_dataset = base_data_pipeline_class.evaluation_data_pipeline(
        evaluation_data=_EVALUATION_DATA
    )
    actual_output = [
        list(evaluation_dataset.take(1))[0]["states"].numpy().tolist(),
    ]
    self.assertEqual(actual_output, [[[0.0, 1.0]]])


if __name__ == "__main__":
  absltest.main()
