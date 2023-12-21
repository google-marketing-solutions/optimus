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

"""Tests for utility functions in optimizers.py."""
from absl.testing import absltest
import optax

from optimus.trainer_lib import optimizers


class OptimizersTests(absltest.TestCase):

  def test_get_optimizer(self):
    self.assertEqual(
        optimizers.get_optimizer(optimizer_name="adam"), optax.adam
    )

  def test_get_optimizer_lookup_error(self):
    with self.assertRaisesRegex(LookupError, "Unrecognized optimizer name"):
      optimizers.get_optimizer(optimizer_name="other_optimizer")


if __name__ == "__main__":
  absltest.main()
