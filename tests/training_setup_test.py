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

"""Tests for utility functions in training_setup.py."""
import os

from absl.testing import absltest
from absl.testing import parameterized
import pandas as pd
import tensorflow as tf

from optimus import training_setup


class TrainingSetupTests(parameterized.TestCase):

  def test_create_mapping_for_categorical_dimensions(self):
    dataframe = pd.DataFrame.from_records(
        [{"a": 0.1, "b": 0}, {"a": 0.2, "b": 1}]
    )
    actual_output = training_setup.create_mapping_for_categorical_dimensions(
        dataframe=dataframe, categorical_column_names=["b",]
    )
    self.assertEqual(actual_output, {"b": 2})

  def test_set_experiment_directory(self):
    experiment_directory = self.create_tempdir().full_path
    folder_one_path = os.path.join(experiment_directory, "experiment_directory")
    tf.io.gfile.makedirs(folder_one_path)
    self.assertEqual(
        training_setup.set_experiment_directory(
            experiment_directory=folder_one_path
        ),
        folder_one_path,
    )

  @parameterized.named_parameters(
      dict(
          testcase_name="incorrect_accelerator",
          accelerator="other_accelerator",
          error_message="Accelerator must be one of",
      ),
      dict(
          testcase_name="gpu_no_coordinator_address",
          accelerator="gpu",
          error_message="coordinator_address is required",
      ),
  )
  def test_set_hardware_value_error(self, accelerator, error_message):
    with self.assertRaisesRegex(ValueError, error_message):
      training_setup.set_hardware(accelerator=accelerator)


if __name__ == "__main__":
  absltest.main()
