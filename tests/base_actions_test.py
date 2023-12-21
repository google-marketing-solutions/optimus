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

"""Tests for utility functions in base_actions.py."""
from absl.testing import absltest

from optimus.actions_lib import base_actions


class BaseActionsTests(absltest.TestCase):
  """Tests utility functions in base_actions.py."""

  def test_initialization(self):
    self.assertIsInstance(
        base_actions.BaseActions(
            hyperparameters=base_actions.DEFAULT_HYPERPARAMETERS
        ),
        base_actions.BaseActions,
    )


if __name__ == "__main__":
  absltest.main()
