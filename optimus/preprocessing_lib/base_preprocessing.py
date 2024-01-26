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


def encode_categorical_value(
    *,
    row_value: str | int,
    column_name: str | int,
    category_mapping: Mapping[str | int, Mapping[str | int, int]],
    unknown_categorical_encoding_value: int = -1,
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
  try:
    return category_mapping[column_name][row_value]
  except KeyError:
    return unknown_categorical_encoding_value


def verify_numerical_columns(
    *, dataframe: pd.DataFrame, categorical_columns: Sequence[str | int]
) -> None:
  """Verifies that the numerical columns are of allowed types.

  Args:
    dataframe: A dataframe to preprocess.
    categorical_columns: A sequence with categorical column names.
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
        f" {','.join(_NON_NUMERICAL_PANDAS_DTYPES)}.",
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
  """
  all_categorical_columns = mapping.keys()
  if set(all_categorical_columns) != set(categorical_columns):
    raise ValueError(
        "Categorical columns mappings must have keys representing each"
        " categorical column. Currently the keys are"
        f" {all_categorical_columns} and the categorical columns are"
        f" {categorical_columns}."
    )


class BaseDataPreprocessor:
  """A class to preprocess a dataframe before using it with Optimus.

  Attributes:
    dataframe: A dataframe to preprocess.
    categorical_column_threshold: A number of unique column values below which a
      column will be classified as categorical. All the other columns will be
      understood as numerical.
    categorical_columns: A sequence with categorical column names.
    categorical_columns_indexes: A sequence with categorical column indexes.
    categorical_columns_unique_values: A mapping between categorical columns and
      their unique values.
    categorical_columns_dimensions: A seuqnce with the number of unique values
      per categorical column
    categories_mappings: A mapping of categorical columns to mappings of their
      unique values and representative integers.
    preprocessed_array: An array with all the preprocessed dataframe values.
    minimum_categorical_encoding_value: A minimum numerical value to use for
      encoding. I.e. if 1 then the encoded values will begin at 1 and and on
      number_of_unique_values + 1.
    unknown_categorical_encoding_value: An integer to use when trying to encode
      an unknown categorical value.
  """

  def __init__(
      self,
      *,
      dataframe: pd.DataFrame,
      skip_columns: Sequence[str | int] | None = None,
      override_categorical_columns: Sequence[str | int] | None = None,
      categorical_column_threshold: int = 200,
      override_categorical_columns_unique_values: (
          Mapping[str | int, Sequence[str | int]] | None
      ) = None,
      categorical_columns_unique_values_path: str | None = None,
      override_categorical_columns_encoding_mapping: (
          Mapping[str | int, Mapping[str | int, int]] | None
      ) = None,
      categorical_columns_encoding_mapping_path: str | None = None,
      minimum_categorical_encoding_value: int = 1,
      unknown_categorical_encoding_value: int = -1,
  ):
    """Initializes the BaseDataPreprocessor class.

    Args:
      dataframe: A dataframe to preprocess.
      skip_columns: A sequence with column names that should not be included in
        the output.
      override_categorical_columns: A sequence with categorical column names.
      categorical_column_threshold: A number of unique column values below which
        a column will be classified as categorical. All the other columns will
        be understood as numerical.
      override_categorical_columns_unique_values: A mapping between column names
        and their unique values.
      categorical_columns_unique_values_path: A path to a pickle file with a
        mapping between column names and their unique values. The
        `override_categorical_columns_unique_values` argument has priority if
        both are provided.
      override_categorical_columns_encoding_mapping: A mapping of categorical
        column names to mappings between their unique and encoded values.
      categorical_columns_encoding_mapping_path: A path to a pickle file with a
        mapping of categorical columns to mappings of their unique values and
        representative integers.
      minimum_categorical_encoding_value: A minimum numerical value to use for
        encoding. I.e. if 1 then the encoded values will begin at 1 and and on
        number_of_unique_values + 1.
      unknown_categorical_encoding_value: An integer to use when trying to
        encode an unknown categorical value.
    """
    self.dataframe = (
        dataframe.drop(columns=skip_columns, inplace=False)
        if skip_columns
        else dataframe
    )
    self._override_categorical_columns = override_categorical_columns
    self.categorical_column_threshold = categorical_column_threshold
    self._override_categorical_columns_unique_values = (
        override_categorical_columns_unique_values
    )
    self._categorical_columns_unique_values_path = (
        categorical_columns_unique_values_path
    )
    self._override_categorical_columns_encoding_mapping = (
        override_categorical_columns_encoding_mapping
    )
    self._categorical_columns_encoding_mapping_path = (
        categorical_columns_encoding_mapping_path
    )
    self.minimum_categorical_encoding_value = minimum_categorical_encoding_value
    self.unknown_categorical_encoding_value = unknown_categorical_encoding_value

  @functools.cached_property
  def categorical_columns(self) -> Sequence[str | int]:
    """Returns a sequence with categorical column names."""
    if self._override_categorical_columns:
      missing_categorical_columns = list(
          sorted(
              set(self._override_categorical_columns)
              - set(self.dataframe.columns.tolist())
          )
      )
      if missing_categorical_columns:
        raise ValueError(
            "The categorical columns:"
            f" {','.join(missing_categorical_columns)} are not present in the"
            " dataframe.",
        )
      verify_numerical_columns(
          dataframe=self.dataframe,
          categorical_columns=self._override_categorical_columns,
      )
      return self._override_categorical_columns
    unique_count = self.dataframe.nunique()
    categorical_columns = unique_count[
        unique_count < self.categorical_column_threshold
    ].index.tolist()
    verify_numerical_columns(
        dataframe=self.dataframe, categorical_columns=categorical_columns
    )
    return categorical_columns

  @functools.cached_property
  def categorical_columns_indexes(self) -> Sequence[int]:
    """Returns a sequence with categorical column indexes."""
    return [
        self.dataframe.columns.tolist().index(categorical_column)
        for categorical_column in self.categorical_columns
    ]

  @functools.cached_property
  def categorical_columns_unique_values(
      self,
  ) -> Mapping[str | int, Sequence[str | int]]:
    """Returns a mapping between categorical columns and their unique values."""
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
    categorical_dataframe = self.dataframe[self.categorical_columns].astype(
        "category"
    )
    return {
        column_name: column_values.cat.categories.values.tolist()
        for column_name, column_values in categorical_dataframe.items()
    }

  @functools.cached_property
  def categorical_columns_dimensions(self) -> Sequence[int]:
    """Returns the number of unique values per categorical column."""
    return [
        len(self.categorical_columns_unique_values[categorical_column])
        for categorical_column in self.categorical_columns
    ]

  @functools.cached_property
  def categories_mappings(
      self,
  ) -> Mapping[str | int, Mapping[str | int, int]]:
    """Returns a mapping of categorical columns, unique and encoded values."""
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
    partial_create_category_to_number_mapping = functools.partial(
        create_category_to_number_mapping,
        minimum_categorical_encoding_value=self.minimum_categorical_encoding_value,
    )
    return {
        key: partial_create_category_to_number_mapping(
            category_unique_values=values,
        )
        for key, values in self.categorical_columns_unique_values.items()
    }

  @functools.cached_property
  def _encoded_categorical_columns(self) -> np.ndarray:
    """Returns an array with encoded categorical values from the dataframe."""
    vectorized_encoder = np.vectorize(
        functools.partial(
            encode_categorical_value,
            category_mapping=self.categories_mappings,
            unknown_categorical_encoding_value=self.unknown_categorical_encoding_value,
        )
    )
    return vectorized_encoder(
        row_value=self.dataframe[self.categorical_columns].values,
        column_name=self.categorical_columns,
    )

  @functools.cached_property
  def preprocessed_array(self) -> np.ndarray:
    """Returns an array with all the preprocessed dataframe values."""
    data_array = self.dataframe.to_numpy()
    data_array[:, self.categorical_columns_indexes] = (
        self._encoded_categorical_columns
    )
    return data_array.astype(np.float32)
