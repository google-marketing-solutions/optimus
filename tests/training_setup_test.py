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

from absl.testing import absltest
from absl.testing import parameterized

from optimus import training_setup


class TrainingSetupTests(parameterized.TestCase):

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
