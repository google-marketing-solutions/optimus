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

"""Tests for utility functions data_pipelines.py."""
from absl.testing import absltest

from optimus.data_pipeline_lib import base_data_pipeline
from optimus.data_pipeline_lib import data_pipelines


class DataPipelinesTests(absltest.TestCase):

  def test_get_data_pipeline(self):
    self.assertEqual(
        data_pipelines.get_data_pipeline(
            data_pipeline_name="base_data_pipeline"
        ),
        base_data_pipeline.BaseDataPipeline,
    )

  def test_get_data_pipeline_value_error(self):
    with self.assertRaisesRegex(
        LookupError, "Unrecognized data pipeline class name"
    ):
      data_pipelines.get_data_pipeline(data_pipeline_name="other_data_pipeline")

  def test_get_base_data_hyperparameters(self):
    self.assertEqual(
        data_pipelines.get_data_pipeline_hyperparameters(
            data_pipeline_name="base_data_pipeline"
        ),
        base_data_pipeline.DEFAULT_HYPERPARAMETERS,
    )

  def test_get_base_data_hyperparameters_value_error(self):
    with self.assertRaisesRegex(
        LookupError, "Unrecognized data pipeline class name"
    ):
      data_pipelines.get_data_pipeline_hyperparameters(
          data_pipeline_name="other_data_pipeline"
      )


if __name__ == "__main__":
  absltest.main()
