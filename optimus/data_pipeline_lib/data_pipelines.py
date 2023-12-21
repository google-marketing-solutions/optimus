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

"""Registry of all the available data pipelines."""
import dataclasses

from ml_collections.config_dict import config_dict

from optimus.data_pipeline_lib import base_data_pipeline


@dataclasses.dataclass(frozen=True)
class _DataPipeline:
  """Instance with and a data pipeline class name, its instance and hyperparameters.

  Attributes:
    name: A name of a data pipeline class.
    instance: An instance of a data pipeline class.
    hyperparameters: Hyperparameters of a data pipeline class.
  """

  name: str
  instance: type[base_data_pipeline.BaseDataPipeline]
  hyperparameters: config_dict.ConfigDict


_ALL_DATA_PIPELINES = config_dict.ConfigDict(
    dict(
        base_data_pipeline=_DataPipeline(
            "base_data_pipeline",
            base_data_pipeline.BaseDataPipeline,
            base_data_pipeline.DEFAULT_HYPERPARAMETERS,
        )
    )
)


def get_data_pipeline(
    *, data_pipeline_name: str = "base_data_pipeline"
) -> type[base_data_pipeline.BaseDataPipeline]:
  """Maps the data pipeline name with the corresponding data pipeline class.

  Args:
    data_pipeline_name: Data pipeline name.

  Returns:
    The requested data pipeline class.

  Raises:
    LookupError: An error when trying to access an unavailable data pipeline
    class.
  """
  if data_pipeline_name not in _ALL_DATA_PIPELINES:
    raise LookupError(
        f"Unrecognized data pipeline class name: {data_pipeline_name}"
    )
  return _ALL_DATA_PIPELINES[data_pipeline_name].instance


def get_data_pipeline_hyperparameters(
    *, data_pipeline_name: str = "base_data_pipeline"
) -> config_dict.ConfigDict:
  """Maps the data pipeline name with the corresponding data pipeline hyperparameters.

  Args:
    data_pipeline_name: Data pipeline name.

  Returns:
    The requested data pipeline class hyperparameters.

  Raises:
    LookupError: An error when trying to access an unavailable data pipeline
    class.
  """
  if data_pipeline_name not in _ALL_DATA_PIPELINES:
    raise LookupError(
        f"Unrecognized data pipeline class name: {data_pipeline_name}"
    )
  return _ALL_DATA_PIPELINES[data_pipeline_name].hyperparameters
