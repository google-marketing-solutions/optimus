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
from absl.testing import parameterized
import numpy as np
import pandas as pd
from optimus.preprocessing_lib import base_preprocessing

_TEST_DATAFRAME = pd.DataFrame({
    "a": [1, 1, 3],
    "b": [4, 4, 6],
    "c": [7, 8, 9],
    "d": [10, 11, 12],
})


class BasePreprocessingUtilityFunctionsTests(parameterized.TestCase):

  def test_create_category_to_number_mapping(self):
    actual_output = base_preprocessing.create_category_to_number_mapping(
        category_unique_values=["a", "b", "c", "d"],
    )
    self.assertEqual(actual_output, {"a": 1, "b": 2, "c": 3, "d": 4})

  @parameterized.named_parameters(
      dict(
          testcase_name="known_value",
          row_value="a",
          result=1,
      ),
      dict(
          testcase_name="unknown_value",
          row_value="b",
          result=-1,
      ),
  )
  def test_encode_categorical_value(self, row_value, result):
    actual_output = base_preprocessing.encode_categorical_value(
        row_value=row_value,
        column_name="column",
        category_mapping={"column": {"a": 1}},
    )
    self.assertEqual(actual_output, result)

  def test_verify_numerical_columns(self):
    with self.assertRaisesRegex(TypeError, "columns are of unallowed types"):
      base_preprocessing.verify_numerical_columns(
          dataframe=pd.DataFrame({
              "a": np.asarray([1, 1, 3], dtype=object),
              "b": np.asarray([4, 4, 6], dtype=np.int32),
          }),
          categorical_columns="b",
      )


class BasePreprocessingTests(parameterized.TestCase):

  def test_skip_columns(self):
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        dataframe=_TEST_DATAFRAME,
        skip_columns=["c", "d"],
    )
    self.assertEqual(base_preprocessor.dataframe.columns.tolist(), ["a", "b"])

  @parameterized.named_parameters(
      dict(
          testcase_name="override_catergorical_columns",
          override_categorical_columns=["a"],
          result=["a"],
      ),
      dict(
          testcase_name="no_override_catergorical_columns",
          override_categorical_columns=None,
          result=["a", "b"],
      ),
  )
  def test_categorical_columns(self, override_categorical_columns, result):
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        dataframe=_TEST_DATAFRAME,
        override_categorical_columns=override_categorical_columns,
        categorical_column_threshold=3,
    )
    self.assertEqual(base_preprocessor.categorical_columns, result)

  def test_categorical_columns_value_error(self):
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        dataframe=_TEST_DATAFRAME,
        override_categorical_columns=["e"],
    )
    with self.assertRaisesRegex(ValueError, "are not present"):
      _ = base_preprocessor.categorical_columns

if __name__ == "__main__":
  absltest.main()
