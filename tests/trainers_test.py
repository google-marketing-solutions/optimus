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

"""Tests for utility functions in the trainers.py."""
from absl.testing import absltest
from absl.testing import parameterized

from optimus.trainer_lib import base_trainer
from optimus.trainer_lib import trainers


class TrainersTests(parameterized.TestCase):

  def test_get_trainer(self):
    self.assertEqual(
        trainers.get_trainer(trainer_name="base_trainer"),
        base_trainer.BaseTrainer,
    )

  def test_get_trainer_value_error(self):
    with self.assertRaisesRegex(LookupError, "Unrecognized trainer class name"):
      trainers.get_trainer(trainer_name="other_trainer")

  def test_get_trainer_hyperparameters(self):
    self.assertEqual(
        trainers.get_trainer_hyperparameters(trainer_name="base_trainer"),
        base_trainer.DEFAULT_HYPERPARAMETERS,
    )

  def test_get_trainer_hyperparameters_lookup_error(self):
    with self.assertRaisesRegex(LookupError, "Unrecognized trainer class name"):
      trainers.get_trainer_hyperparameters(trainer_name="other_trainer")


if __name__ == "__main__":
  absltest.main()
