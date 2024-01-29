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
import os
import pickle
from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import pandas as pd
import tensorflow as tf
from optimus.preprocessing_lib import base_preprocessing

_TEST_DATAFRAME = pd.DataFrame({
    "a": [1, 1, 3],
    "b": [4, 4, 6],
    "c": [7, 8, 9],
    "d": [10, 11, 12],
})
_TEST_COLUMNS = ["a", "b", "c", "d"]


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

  def test_encode_categorical_columns(self):
    actual_output = base_preprocessing.encode_categorical_columns(
        categorical_array=np.asarray(["a", "b", "c"]),
        categorical_columns=["column"],
        categories_mappings={"column": {"a": 1, "b": 2}},
    ).tolist()
    self.assertEqual(actual_output, [1, 2, -1])

  def test_find_categorical_columns_from_dataframe(self):
    actual_output = base_preprocessing.find_categorical_columns_from_dataframe(
        dataframe=_TEST_DATAFRAME,
        categorical_column_threshold=3,
    )
    self.assertEqual(actual_output, ["a", "b"])

  def test_find_categorical_columns_unique_values_from_dataframe(self):
    actual_output = base_preprocessing.find_categorical_columns_unique_values_from_dataframe(
        dataframe=_TEST_DATAFRAME,
        categorical_columns=["a", "b"],
    )
    self.assertEqual(actual_output, {"a": [1, 3], "b": [4, 6]})

  def test_create_categories_mappings(self):
    actual_output = base_preprocessing.create_categories_mappings(
        categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
    )
    self.assertEqual(actual_output, {"a": {1: 1, 3: 2}, "b": {4: 1, 6: 2}})

  def test_verify_override_mapping(self):
    with self.assertRaisesRegex(ValueError, "columns mappings must have keys"):
      base_preprocessing.verify_override_mapping(
          mapping={"a": [1, 3], "b": [4, 6]},
          categorical_columns=["a", "b", "c"],
      )

  def test_check_missing_arguments_unique_values(self):
    with self.assertRaisesRegex(
        ValueError,
        "`categorical_columns_unique_values` or"
        " `categorical_columns_unique_values_path` must be provided",
    ):
      base_preprocessing.check_missing_arguments(
          categorical_columns=["a", "b"],
          categorical_columns_unique_values=None,
          categorical_columns_unique_values_path=None,
          categorical_columns_encoding_mapping={
              "a": {1: 1, 3: 2},
              "b": {4: 1, 6: 2},
          },
          categorical_columns_encoding_mapping_path=None,
      )

  def test_check_missing_arguments_encoding_mapping(self):
    with self.assertRaisesRegex(
        ValueError,
        "`categorical_columns_encoding_mapping` or"
        " `categorical_columns_encoding_mapping_path` must be provided",
    ):
      base_preprocessing.check_missing_arguments(
          categorical_columns=["a", "b"],
          categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
          categorical_columns_unique_values_path=None,
          categorical_columns_encoding_mapping=None,
          categorical_columns_encoding_mapping_path=None,
      )


class BasePreprocessingTests(parameterized.TestCase):

  def test_categorical_columns(self):
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        columns=_TEST_COLUMNS,
        categorical_columns=["a", "b"],
        categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
        categorical_columns_encoding_mapping={
            "a": {1: 1, 3: 2},
            "b": {4: 1, 6: 2},
        },
    )
    self.assertEqual(base_preprocessor.categorical_columns, ["a", "b"])

  def test_categorical_columns_value_error(self):
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        columns=_TEST_COLUMNS,
        categorical_columns=["a", "f"],
        categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
        categorical_columns_encoding_mapping={
            "a": {1: 1, 3: 2},
            "b": {4: 1, 6: 2},
        },
    )
    with self.assertRaisesRegex(ValueError, "are not present"):
      _ = base_preprocessor.categorical_columns

  @parameterized.named_parameters(
      dict(
          testcase_name="with_skip_columns",
          skip_columns=["a"],
          result=[0],
      ),
      dict(
          testcase_name="without_skip_columns",
          skip_columns=None,
          result=[0, 1],
      ),
  )
  def test_categorical_columns_indexes(self, skip_columns, result):
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        columns=_TEST_COLUMNS,
        skip_columns=skip_columns,
        categorical_columns=["a", "b"],
        categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
        categorical_columns_encoding_mapping={
            "a": {1: 1, 3: 2},
            "b": {4: 1, 6: 2},
        },
    )
    self.assertEqual(
        base_preprocessor.categorical_columns_indexes, result
    )

  def test_categorical_columns_unique_values_from_file(self):
    mapping = {"a": [1, 3], "b": [4, 6]}
    file_path = os.path.join(self.create_tempdir().full_path, "mapping.pickle")
    with tf.io.gfile.GFile(file_path, "wb") as file:
      pickle.dump(mapping, file)
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        columns=_TEST_COLUMNS,
        categorical_columns=["a", "b"],
        categorical_columns_unique_values_path=file_path,
        categorical_columns_encoding_mapping={
            "a": {1: 1, 3: 2},
            "b": {4: 1, 6: 2},
        },
    )
    self.assertEqual(
        base_preprocessor.categorical_columns_unique_values, mapping
    )

  def test_categorical_columns_unique_values(self):
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        columns=_TEST_COLUMNS,
        categorical_columns=["a", "b"],
        categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
        categorical_columns_encoding_mapping={
            "a": {1: 1, 3: 2},
            "b": {4: 1, 6: 2},
        },
    )
    self.assertEqual(
        base_preprocessor.categorical_columns_unique_values,
        {"a": [1, 3], "b": [4, 6]},
    )

  def test_categorical_columns_dimensions(self):
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        columns=_TEST_COLUMNS,
        categorical_columns=["a", "b"],
        categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
        categorical_columns_encoding_mapping={
            "a": {1: 1, 3: 2},
            "b": {4: 1, 6: 2},
        },
    )
    self.assertEqual(base_preprocessor.categorical_columns_dimensions, [2, 2])

  def test_categories_mappings_from_file(self):
    mapping = {"a": {1: 1, 3: 2}, "b": {4: 1, 6: 2}}
    file_path = os.path.join(self.create_tempdir().full_path, "mapping.pickle")
    with tf.io.gfile.GFile(file_path, "wb") as file:
      pickle.dump(mapping, file)
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        columns=_TEST_COLUMNS,
        categorical_columns=["a", "b"],
        categorical_columns_encoding_mapping_path=file_path,
        categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
    )
    self.assertEqual(base_preprocessor.categories_mappings, mapping)

  def test_categories_mappings(self):
    mapping = {"a": {1: 1, 3: 2}, "b": {4: 1, 6: 2}}
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        columns=_TEST_COLUMNS,
        categorical_columns=["a", "b"],
        categorical_columns_encoding_mapping=mapping,
        categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
    )
    self.assertEqual(
        base_preprocessor.categories_mappings,
        mapping,
    )

  def test_preprocess_data(self):
    base_preprocessor = base_preprocessing.BaseDataPreprocessor(
        columns=_TEST_COLUMNS,
        categorical_columns=["a", "b"],
        skip_columns=["d"],
        categorical_columns_unique_values={"a": [1, 3], "b": [4, 6]},
        categorical_columns_encoding_mapping={
            "a": {1: 1, 3: 2},
            "b": {4: 1, 6: 2},
        },
    )
    actual_output = base_preprocessor.preprocess_data(
        input_data=_TEST_DATAFRAME
    ).tolist()
    expected_output = [[1.0, 1.0, 7.0], [1.0, 1.0, 8.0], [2.0, 2.0, 9.0]]
    self.assertEqual(actual_output, expected_output)


if __name__ == "__main__":
  absltest.main()
