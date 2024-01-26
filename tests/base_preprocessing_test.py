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

"""Tests for utility functions in base_preprocessing.py."""
from absl.testing import absltest
from optimus.preprocessing_lib import base_preprocessing


class BasePreprocessingTests(absltest.TestCase):

  def test_create_category_to_number_mapping(self):
    actual_output = base_preprocessing.create_category_to_number_mapping(
        category_unique_values=["a", "b", "c", "d"],
    )
    self.assertEqual(actual_output, {"a": 1, "b": 2, "c": 3, "d": 4})


if __name__ == "__main__":
  absltest.main()
