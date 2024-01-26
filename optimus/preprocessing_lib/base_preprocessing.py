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

"""Base preprocessing class."""
from typing import Mapping, Sequence


def create_category_to_number_mapping(
    *,
    category_unique_values: Sequence[str | int],
    minimum_categorical_encoding_value: int = 1,
) -> Mapping[str | int, int]:
  """Returns a mapping between unique column values and numerical values.

  Args:
    category_unique_values: A sequence with unique values from a categorical
      column.
    minimum_categorical_encoding_value: A minimum numerical value to use for
      encoding. I.e. if 1 then the encoded values will begin at 1 and and on
      number_of_unique_values + 1.
  """
  encoding_range = range(
      minimum_categorical_encoding_value,
      len(category_unique_values) + minimum_categorical_encoding_value,
  )
  return dict(zip(category_unique_values, encoding_range))
