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
from typing import Final, List, Mapping, Sequence
import pandas as pd

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


class BaseDataPreprocessor:
  """A class to preprocess a dataframe before using it with Optimus.

  Attributes:
    dataframe: A dataframe to preprocess.
    categorical_column_threshold: A number of unique column values below which a
      column will be classified as categorical. All the other columns will be
      understood as numerical.
    categorical_columns: A sequence with categorical column names.
    categorical_columns_indexes: A sequence with categorical column indexes.
  """

  def __init__(
      self,
      *,
      dataframe: pd.DataFrame,
      skip_columns: Sequence[str | int] | None = None,
      override_categorical_columns: Sequence[str | int] | None = None,
      categorical_column_threshold: int = 200,
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
    """
    self.dataframe = (
        dataframe.drop(columns=skip_columns, inplace=False)
        if skip_columns
        else dataframe
    )
    self._override_categorical_columns = override_categorical_columns
    self.categorical_column_threshold = categorical_column_threshold

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
