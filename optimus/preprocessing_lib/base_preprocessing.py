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
import functools
import logging
import pickle
from typing import Final, List, Mapping, Sequence
import numpy as np
import pandas as pd
import tensorflow as tf

_NON_NUMERICAL_PANDAS_DTYPES: Final[List[str]] = [
    "object",
    "bool",
    "datetime",
    "timedelta",
    "category",
    "datetimetz",
]
_NAN_SUBSTITUTE: Final[float] = 0.0


def create_category_to_number_mapping(
    *,
    category_unique_values: Sequence[str | int],
    minimum_categorical_encoding_value: int = 1,
    classes_encoding: bool = False,
) -> Mapping[str | int, int] | Mapping[int, str | int]:
  """Returns a mapping between unique column values and numerical values.

  Args:
    category_unique_values: A sequence with unique values from a categorical
      column.
    minimum_categorical_encoding_value: A minimum numerical value to use for
      encoding. I.e. if 1 then the encoded values will begin at 1 and and on
      number_of_unique_values + 1.
    classes_encoding: An indicator whether the encoding is for the features or
      classes.
  """
  encoding_range = range(
      minimum_categorical_encoding_value,
      len(category_unique_values) + minimum_categorical_encoding_value,
  )
  if not classes_encoding:
    return dict(zip(category_unique_values, encoding_range))
  return dict(zip(encoding_range, category_unique_values))


def encode_categorical_value(
    *,
    row_value: str | int,
    column_name: str | int,
    category_mapping: Mapping[str | int, Mapping[str | int, int]],
    unknown_categorical_encoding_value: int = 0,
) -> int:
  """Returns a encoded categorical value.

  Args:
    row_value: A categorical value to be encoded.
    column_name: A categorical column name.
    category_mapping: A mapping between categorical column names and their
      unique values mapped to integers.
    unknown_categorical_encoding_value: An integer to use when trying to encode
      an unknown categorical value.
  """
  column_mapping = category_mapping[column_name]
  try:
    return column_mapping[row_value]
  except KeyError:
    return unknown_categorical_encoding_value


def encode_categorical_columns(
    *,
    categorical_array: np.ndarray,
    categorical_columns: Sequence[str | int],
    categories_mappings: Mapping[str | int, Mapping[str | int, int]],
    unknown_categorical_encoding_value: int = 0,
) -> np.ndarray:
  """Returns an array with encoded categorical values from the dataframe.

  Args:
    categorical_array: An array with categorical columns only to preprocess.
    categorical_columns: A sequence with categorical column names.
    categories_mappings: A mapping between categorical column names and their
      unique values mapped to integers.
    unknown_categorical_encoding_value: An integer to use when trying to encode
      an unknown categorical value.
  """
  vectorized_encoder = np.vectorize(
      functools.partial(
          encode_categorical_value,
          category_mapping=categories_mappings,
          unknown_categorical_encoding_value=unknown_categorical_encoding_value,
      )
  )
  return vectorized_encoder(
      row_value=categorical_array,
      column_name=categorical_columns,
  )


def verify_numerical_columns(
    *, dataframe: pd.DataFrame, categorical_columns: Sequence[str | int]
) -> None:
  """Verifies that the numerical columns are of allowed types.

  Args:
    dataframe: A dataframe to preprocess.
    categorical_columns: A sequence with categorical column names.

  Raises:
    TypeError: An error when the determined numerical column data types are not
    allowed.
  """
  numerical_columns = list(
      sorted(set(dataframe.columns.tolist()) - set(categorical_columns))
  )
  unallowed_numerical_dtypes = (
      dataframe[numerical_columns]
      .select_dtypes(include=_NON_NUMERICAL_PANDAS_DTYPES)
      .dtypes.tolist()
  )
  if unallowed_numerical_dtypes:
    raise TypeError(
        "The specified / detected numerical columns are of unallowed types:"
        f" {', '.join(_NON_NUMERICAL_PANDAS_DTYPES)}.",
    )


def verify_override_mapping(
    *,
    mapping: (
        Mapping[str | int, Sequence[str | int]]
        | Mapping[str | int, Mapping[str | int, int]]
    ),
    categorical_columns: Sequence[str | int],
) -> None:
  """Verifies that the mapping has keys represeting each categorical column.

  Args:
    mapping: A mapping between categorical columns and their unique values. Or a
      mapping of categorical columns to mappings of their unique values and
      representative integers.
    categorical_columns: A sequence with categorical column names.

  Raises:
    ValueError: An error when the categorical column name keys are not the same
    as the provided sequence with the categorical column names.
  """
  all_categorical_columns = mapping.keys()
  if set(all_categorical_columns) != set(categorical_columns):
    raise ValueError(
        "Categorical columns mappings must have keys representing each"
        " categorical column. Currently the keys are"
        f" {all_categorical_columns} and the categorical columns are"
        f" {categorical_columns}."
    )


def find_categorical_columns_from_dataframe(
    dataframe: pd.DataFrame, categorical_column_threshold: int = 200
) -> Sequence[str | int]:
  """Returns a sequence with categorical column names.

  Args:
    dataframe: A dataframe with all the columns.
    categorical_column_threshold: A number of unique column values below which a
      column will be classified as categorical. All the other columns will be
      understood as numerical.
  """
  unique_count = dataframe.nunique()
  categorical_columns = unique_count[
      unique_count < categorical_column_threshold
  ].index.tolist()
  verify_numerical_columns(
      dataframe=dataframe, categorical_columns=categorical_columns
  )
  return categorical_columns


def find_categorical_columns_unique_values_from_dataframe(
    dataframe: pd.DataFrame, categorical_columns: Sequence[str | int]
) -> Mapping[str | int, Sequence[str | int]]:
  """Returns a mapping between categorical columns and their unique values.

  Args:
    dataframe: A dataframe with all the columns.
    categorical_columns: A sequence with categorical column names.
  """
  categorical_dataframe = dataframe[categorical_columns].astype("category")
  return {
      column_name: column_values.cat.categories.values.tolist()
      for column_name, column_values in categorical_dataframe.items()
  }


def create_categories_mappings(
    categorical_columns_unique_values: Mapping[str | int, Sequence[str | int]],
    minimum_categorical_encoding_value: int = 1,
) -> Mapping[str | int, Mapping[str | int, int]]:
  """Returns a mapping of categorical columns to mappings of their unique values and representative integers.

  Args:
    categorical_columns_unique_values: A mapping between categorical columns and
      their unique values.
    minimum_categorical_encoding_value: A minimum numerical value to use for
      encoding. I.e. if 1 then the encoded values will begin at 1 and and on
      number_of_unique_values + 1.
  """
  partial_create_category_to_number_mapping = functools.partial(
      create_category_to_number_mapping,
      minimum_categorical_encoding_value=minimum_categorical_encoding_value,
  )
  return {
      key: partial_create_category_to_number_mapping(
          category_unique_values=values,
      )
      for key, values in categorical_columns_unique_values.items()
  }


def check_missing_arguments(
    *,
    categorical_columns: Sequence[str | int] | None = None,
    categorical_columns_unique_values: (
        Mapping[str | int, Sequence[str | int]] | None
    ) = None,
    categorical_columns_unique_values_path: str | None = None,
    categorical_columns_encoding_mapping: (
        Mapping[str | int, Mapping[str | int, int]] | None
    ) = None,
    categorical_columns_encoding_mapping_path: str | None = None,
) -> None:
  """Checks if all required arguments were provided.

  Args:
    categorical_columns: A sequence with categorical column names.
    categorical_columns_unique_values: A mapping between column names and their
      unique values.
    categorical_columns_unique_values_path: A path to a pickle file with a
      mapping between column names and their unique values. The
      `categorical_columns_unique_values` argument has priority if both are
      provided.
    categorical_columns_encoding_mapping: A mapping of categorical column names
      to mappings between their unique and encoded values.
    categorical_columns_encoding_mapping_path: A path to a pickle file with a
      mapping of categorical columns to mappings of their unique values and
      representative integers.

  Raises:
    ValueError: An error is there are any missing required values for the
    BaseDataPreprocessor class initialization.
  """
  if categorical_columns:
    if (
        not categorical_columns_unique_values
        and not categorical_columns_unique_values_path
    ):
      raise ValueError(
          "`categorical_columns_unique_values` or"
          " `categorical_columns_unique_values_path` must be provided if"
          " `categorical_columns` is provided."
      )
    if (
        not categorical_columns_encoding_mapping
        and not categorical_columns_encoding_mapping_path
    ):
      raise ValueError(
          "`categorical_columns_encoding_mapping` or"
          " `categorical_columns_encoding_mapping_path` must be provided if"
          " `categorical_columns` is provided."
      )


class BaseDataPreprocessor:
  """A class to preprocess a dataframe before using it with Optimus.

  Attributes:
    columns: A sequence with all the columns in the dataframe.
    skip_columns: A sequence with column names that should not be included in
      the output.
    categorical_columns: A sequence with categorical column names.
    categorical_columns_indexes: A sequence with categorical column indexes.
    categorical_columns_unique_values: A mapping between categorical columns and
      their unique values.
    categorical_columns_dimensions: A sequence with the number of unique values
      per categorical column
    categories_mappings: A mapping of categorical columns to mappings of their
      unique values and representative integers.
    output_classes: A sequence with unique labels. I.e. when a model predicts an
      action as a class.
    output_classes_encoding: A mapping between encodings and output classes.
  """

  def __init__(
      self,
      *,
      columns: Sequence[str | int],
      skip_columns: Sequence[str | int] | None = None,
      categorical_columns: Sequence[str | int] | None = None,
      categorical_columns_unique_values: (
          Mapping[str | int, Sequence[str | int]] | None
      ) = None,
      categorical_columns_unique_values_path: str | None = None,
      categorical_columns_encoding_mapping: (
          Mapping[str | int, Mapping[str | int, int]] | None
      ) = None,
      categorical_columns_encoding_mapping_path: str | None = None,
      output_classes: Sequence[int | str] | None = None,
      output_classes_encoding_path: str | None = None,
      action_space: int | None = None,
      unknown_categorical_encoding_value: int = 0,
  ):
    """Initializes the BaseDataPreprocessor class.

    Args:
      columns: A sequence with all the columns in the dataframe.
      skip_columns: A sequence with column names that should not be included in
        the output.
      categorical_columns: A sequence with categorical column names.
      categorical_columns_unique_values: A mapping between column names and
        their unique values.
      categorical_columns_unique_values_path: A path to a pickle file with a
        mapping between column names and their unique values. The
        `categorical_columns_unique_values` argument has priority if both are
        provided.
      categorical_columns_encoding_mapping: A mapping of categorical column
        names to mappings between their unique and encoded values.
      categorical_columns_encoding_mapping_path: A path to a pickle file with a
        mapping of categorical columns to mappings of their unique values and
        representative integers.
      output_classes: A sequence with unique labels. I.e. when a model predicts
        an action as a class.
      output_classes_encoding_path: A path to a pickle file with a mapping
        between unique actions (i.e. classes) and their encodings.
      action_space: A number of actions available to the Optimus model.
      unknown_categorical_encoding_value: An integer to use when trying to
        encode an unknown categorical value.
    """
    check_missing_arguments(
        categorical_columns=categorical_columns,
        categorical_columns_unique_values=categorical_columns_unique_values,
        categorical_columns_unique_values_path=categorical_columns_unique_values_path,
        categorical_columns_encoding_mapping=categorical_columns_encoding_mapping,
        categorical_columns_encoding_mapping_path=categorical_columns_encoding_mapping_path,
    )
    self.columns = columns
    self.skip_columns = skip_columns
    self._override_categorical_columns = categorical_columns
    self._override_categorical_columns_unique_values = (
        categorical_columns_unique_values
    )
    self._categorical_columns_unique_values_path = (
        categorical_columns_unique_values_path
    )
    self._override_categorical_columns_encoding_mapping = (
        categorical_columns_encoding_mapping
    )
    self._categorical_columns_encoding_mapping_path = (
        categorical_columns_encoding_mapping_path
    )
    self._override_output_classes = output_classes
    self._output_classes_encoding_path = output_classes_encoding_path
    self._action_space = action_space
    self._unknown_categorical_encoding_value = (
        unknown_categorical_encoding_value
    )

  @functools.cached_property
  def categorical_columns(self) -> Sequence[str | int] | None:
    """Returns a sequence with categorical column names.

    Raises:
      ValueError: An error when any provided categorical column name is not
      present in the list with all column names.
    """
    if not self._override_categorical_columns:
      return None
    missing_categorical_columns = list(
        sorted(set(self._override_categorical_columns) - set(self.columns))
    )
    if missing_categorical_columns:
      raise ValueError(
          "The categorical columns:"
          f" {', '.join(missing_categorical_columns)} are not present in the"
          " dataframe.",
      )
    return self._override_categorical_columns

  @functools.cached_property
  def categorical_columns_indexes(self) -> Sequence[int] | None:
    """Returns a sequence with categorical column indexes."""
    if not self.categorical_columns:
      return None
    if not self.skip_columns:
      columns = self.columns
      categorical_columns = self.categorical_columns
    else:
      columns = [
          column for column in self.columns if column not in self.skip_columns
      ]
      categorical_columns = [
          column
          for column in self.categorical_columns
          if column not in self.skip_columns
      ]
    return [
        columns.index(categorical_column)
        for categorical_column in categorical_columns
    ]

  @functools.cached_property
  def categorical_columns_unique_values(
      self,
  ) -> Mapping[str | int, Sequence[str | int]] | None:
    """Returns a mapping between categorical columns and their unique values."""
    if not self.categorical_columns:
      return None
    if self._override_categorical_columns_unique_values:
      verify_override_mapping(
          mapping=self._override_categorical_columns_unique_values,
          categorical_columns=self.categorical_columns,
      )
      return self._override_categorical_columns_unique_values
    elif self._categorical_columns_unique_values_path:
      with tf.io.gfile.GFile(
          self._categorical_columns_unique_values_path, "rb"
      ) as artifact:
        mapping = pickle.load(artifact)
        verify_override_mapping(
            mapping=mapping, categorical_columns=self.categorical_columns
        )
        return mapping

  @functools.cached_property
  def categorical_columns_dimensions(self) -> Sequence[int] | None:
    """Returns the number of unique values per categorical column."""
    if not self.categorical_columns:
      return None
    return [
        len(self.categorical_columns_unique_values[categorical_column]) + 1
        for categorical_column in self.categorical_columns
    ]

  @functools.cached_property
  def categories_mappings(
      self,
  ) -> Mapping[str | int, Mapping[str | int, int]] | None:
    """Returns a mapping of categorical columns, unique and encoded values."""
    if not self.categorical_columns:
      return None
    if self._override_categorical_columns_encoding_mapping:
      verify_override_mapping(
          mapping=self._override_categorical_columns_encoding_mapping,
          categorical_columns=self.categorical_columns,
      )
      return self._override_categorical_columns_encoding_mapping
    if self._categorical_columns_encoding_mapping_path:
      with tf.io.gfile.GFile(
          self._categorical_columns_encoding_mapping_path, "rb"
      ) as artifact:
        mapping = pickle.load(artifact)
        verify_override_mapping(
            mapping=mapping, categorical_columns=self.categorical_columns
        )
        return mapping

  @functools.cached_property
  def _skip_columns_indexes(self) -> Sequence[int]:
    """Returns the indexes of the skip columns."""
    if not self.skip_columns:
      return []
    return sorted([self.columns.index(column) for column in self.skip_columns])

  def preprocess_data(
      self,
      *,
      input_data: np.ndarray,
  ) -> np.ndarray:
    """Returns an array with the preprocessed values from the input array.

    Args:
      input_data: An input array to preprocess.

    Raises:
      ValueError: An error when the input data is not a 2-D array.
    """
    if len(input_data.shape) != 2:
      raise ValueError("`input_data` must be a 2-D array.")
    if self._skip_columns_indexes:
      input_data = np.delete(input_data, self._skip_columns_indexes, axis=1)
    if self.categorical_columns:
      encoded_categorical_columns = encode_categorical_columns(
          categorical_array=input_data[:, self.categorical_columns_indexes],
          categorical_columns=self.categorical_columns,
          categories_mappings=self.categories_mappings,
          unknown_categorical_encoding_value=self._unknown_categorical_encoding_value,
      )
      input_data[:, self.categorical_columns_indexes] = (
          encoded_categorical_columns
      )
    output = input_data.astype(np.float32, copy=False)
    if np.isnan(output).any():
      output = np.nan_to_num(output, nan=_NAN_SUBSTITUTE)
      logging.warning(
          "NaN value(s) were detected and replaced with %s.",
          str(_NAN_SUBSTITUTE),
      )
    return output

  @functools.cached_property
  def output_classes_encoding(
      self,
  ) -> Mapping[str | int, int] | Mapping[int, str | int]:
    """Returns a mapping between encodings and output classes.

    Raises:
      ValueError: An error when both `output_classes` or
      `output_classes_encoding_path` are not provided at the class
      initialization. Or if `action_space` is not provided at the class
      initialization.
    """
    if self._override_output_classes:
      output_classes_encoding = create_category_to_number_mapping(
          category_unique_values=self.output_classes,
          minimum_categorical_encoding_value=0,
          classes_encoding=True,
      )
    elif self._output_classes_encoding_path:
      with tf.io.gfile.GFile(
          self._output_classes_encoding_path, "rb"
      ) as artifact:
        output_classes_encoding = pickle.load(artifact)
    else:
      raise ValueError(
          "`output_classes` or `output_classes_encoding_path` must be provided"
          " at the class initialization if you try to access"
          " `output_classes_encoding`."
      )
    if not self._action_space:
      raise ValueError(
          "`action_space` must be provided at the class initialization."
      )
    if len(output_classes_encoding) != self._action_space:
      raise ValueError(
          "The length of the encoding mapping is not the same as the action"
          f" space, {len(output_classes_encoding)} != {self._action_space}."
      )
    return output_classes_encoding

  @functools.cached_property
  def output_classes(self) -> Sequence[int | str] | None:
    if self._override_output_classes:
      return list(sorted(set(self._override_output_classes)))
    return list(sorted(set(self.output_classes_encoding.values())))

  def postprocess_data(self, input_data: np.ndarray) -> list[str | int]:
    """Returns an array with the postprocessed values from the input array.

    Args:
      input_data: An input array to postprocess.
    """
    return np.vectorize(
        lambda x: self.output_classes_encoding[x], otypes=[object]
    )(input_data).tolist()
