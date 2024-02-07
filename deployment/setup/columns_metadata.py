# Copyright 2024 Google LLC.
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

"""Support module to process column metadata."""

import os
import pickle
from typing import Final, Mapping
from absl import app
from ml_collections.config_dict import config_dict
from optimus.preprocessing_lib import base_preprocessing
import tensorflow as tf

_ALL_COLUMNS_KEY: Final[str] = "all_columns"
_CATEGORICAL_COLUMNS_KEY: Final[str] = "categorical_columns"
_COLUMN_METADATA_DESTINATION: Final[str] = os.path.join(
    os.getenv("ARTIFACT_DIRECTORY_PATH"), "column_metadata.pickle"
)
_CATEGORICAL_UNIQUE_VALUES_DESTINATION: Final[str] = os.path.join(
    os.getenv("ARTIFACT_DIRECTORY_PATH"), "categorical_unique_values.pickle"
)
_CATEGORICAL_VALUES_ENCODING_DESTINATION: Final[str] = os.path.join(
    os.getenv("ARTIFACT_DIRECTORY_PATH"), "categorical_values_encoding.pickle"
)
_OUTPUT_CLASSES_ENCODING_DESTINATION: Final[str] = os.path.join(
    os.getenv("ARTIFACT_DIRECTORY_PATH"), "output_classes_encoding.pickle"
)


def load_column_metadata() -> config_dict.ConfigDict:
  """Returns column metadata from the passed string.

  The mapping is also saved as a pickle fie in the artifact directory.

  Raises:
      ValueError: An error if the provided metadata has no `all_columns` or
      `categorical_columns` key.
  """
  with tf.io.gfile.GFile(os.getenv("COLUMN_METADATA_PATH"), "rb") as artifact:
    column_metadata = config_dict.ConfigDict(pickle.loads(artifact.read()))
  if _ALL_COLUMNS_KEY not in column_metadata:
    raise ValueError(
        f"The input column metadata has no `{_ALL_COLUMNS_KEY}` key."
    )
  if _CATEGORICAL_COLUMNS_KEY not in column_metadata:
    raise ValueError(
        f"The input column metadata has no `{_CATEGORICAL_COLUMNS_KEY}` key."
    )
  with tf.io.gfile.GFile(_COLUMN_METADATA_DESTINATION, "wb") as artifact:
    pickle.dump(column_metadata.to_dict(), artifact)
  return column_metadata


def load_unique_categorical_values(
    *,
    column_metadata: config_dict.ConfigDict,
) -> Mapping[str | int, str]:
  """Returns a mapping between categorical columns and their unique values.

  The mapping is also saved as a pickle fie in the artifact directory.

  Args:
    column_metadata: A mapping between column categories and column names.

  Raises:
    ValueError: An error when the categorical column name keys are not the same
    as the provided sequence with the categorical column names.
  """
  with tf.io.gfile.GFile(
      os.getenv("CATEGORICAL_UNIQUE_VALUES_PATH"), "rb"
  ) as artifact:
    categorical_columns_unique_values = pickle.loads(artifact.read())
  base_preprocessing.verify_override_mapping(
      mapping=categorical_columns_unique_values,
      categorical_columns=column_metadata.get(_CATEGORICAL_COLUMNS_KEY),
  )
  with tf.io.gfile.GFile(
      _CATEGORICAL_UNIQUE_VALUES_DESTINATION, "wb"
  ) as artifact:
    pickle.dump(categorical_columns_unique_values, artifact)
  return categorical_columns_unique_values


def load_categorical_columns_encoding_mapping(
    *, unique_categorical_values: Mapping[str | int, str]
) -> None:
  """Creates a mapping of categorical columns to mappings of their unique values and representative integers.

  The mapping is also saved as a pickle fie in the artifact directory.

  Args:
    unique_categorical_values: A mapping between categorical columns and their
      unique values.
  """
  categorical_columns_encoding_mapping = (
      base_preprocessing.create_categories_mappings(
          categorical_columns_unique_values=unique_categorical_values
      )
  )
  with tf.io.gfile.GFile(
      _CATEGORICAL_VALUES_ENCODING_DESTINATION, "wb"
  ) as artifact:
    pickle.dump(categorical_columns_encoding_mapping, artifact)


def load_output_classes_encoding() -> None:
  """Creates a mapping between encodings and output classes.

  The mapping is also saved as a pickle fie in the artifact directory.
  """
  with tf.io.gfile.GFile(os.getenv("OUTPUT_CLASSES_PATH"), "rb") as artifact:
    output_classes = pickle.loads(artifact.read())
  output_classes_encoding = (
      base_preprocessing.create_category_to_number_mapping(
          category_unique_values=output_classes,
          minimum_categorical_encoding_value=0,
          classes_encoding=True,
      )
  )
  with tf.io.gfile.GFile(
      _OUTPUT_CLASSES_ENCODING_DESTINATION, "wb"
  ) as artifact:
    pickle.dump(output_classes_encoding, artifact)


def create_column_artifacts(_) -> None:
  """Creates all the column artifacts necessary for data preprocessing."""
  column_metadata = load_column_metadata()
  unique_categorical_values = load_unique_categorical_values(
      column_metadata=column_metadata,
  )
  load_categorical_columns_encoding_mapping(
      unique_categorical_values=unique_categorical_values
  )
  load_output_classes_encoding()


if __name__ == "__main__":
  app.run(create_column_artifacts)
