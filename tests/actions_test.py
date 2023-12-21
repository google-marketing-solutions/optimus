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

"""Tests for utility functions in actions.py."""
from absl.testing import absltest

from optimus.actions_lib import actions
from optimus.actions_lib import base_actions


class ActionsTests(absltest.TestCase):

  def test_get_actions(self):
    self.assertEqual(
        actions.get_actions(actions_name="base_actions"),
        base_actions.BaseActions,
    )

  def test_get_actions_lookup_error(self):
    with self.assertRaisesRegex(LookupError, "Unrecognized actions class name"):
      actions.get_actions(actions_name="other_actions")

  def test_get_actions_hyperparameters(self):
    self.assertEqual(
        actions.get_actions_hyperparameters(actions_name="base_actions"),
        base_actions.DEFAULT_HYPERPARAMETERS,
    )

  def test_get_actions_hyperparameters_lookup_error(self):
    with self.assertRaisesRegex(LookupError, "Unrecognized actions class name"):
      actions.get_actions_hyperparameters(actions_name="other_actions")


if __name__ == "__main__":
  absltest.main()
