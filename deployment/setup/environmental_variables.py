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

"""Support module to process environmental variables."""

import os
from typing import Final, Mapping
from absl import app
from absl import flags
import tensorflow as tf

_USER_ENVIRONMENTAL_VARIABLES_PATH = flags.DEFINE_string(
    "user_environmental_variables_path",
    None,
    "A path to a text file with user environmental variables.",
)
_ARTIFACT_REGISTRY_NAME: Final[str] = (
    "ARTIFACT_REGISTRY_NAME=optimus-artifact-registry"
)
_CLOUD_BUILD_TRIGGER_NAME: Final[str] = (
    "CLOUD_BUILD_TRIGGER_NAME=optimus-build-trigger"
)
_CLOUD_RUN_DOCKER_IMAGE_NAME: Final[str] = (
    "CLOUD_RUN_DOCKER_IMAGE_NAME=optimus_server:latest"
)
_CLOUD_RUN_APP_HOME_DIRECTORY: Final[str] = (
    "CLOUD_RUN_APP_HOME_DIRECTORY=cloud_run"
)
_VERTEX_PIPELINE_DOCKER_IMAGE_NAME: Final[str] = (
    "VERTEX_PIPELINE_DOCKER_IMAGE_NAME=optimus_pipeline:latest"
)
_VERTEX_PIPELINE_NAME: Final[str] = "VERTEX_PIPELINE_NAME=optimus-pipeline"
_CLOUD_RUN_SERVICE_NAME: Final[str] = "CLOUD_RUN_SERVICE_NAME=optimus-server"
_CLOUD_FUNCTION_NAME: Final[str] = (
    "CLOUD_FUNCTION_NAME=optimus_pipeline_trigger"
)
_CLOUD_SOURCE_REPOSITORY_NAME: Final[str] = (
    "CLOUD_SOURCE_REPOSITORY_NAME=optimus_server_repository"
)
_VERTEX_PIPELINE_LOCAL_DIRECTORY_NAME: Final[str] = (
    "VERTEX_PIPELINE_LOCAL_DIRECTORY_NAME=vertex_pipeline"
)
_CLOUD_FUNCTION_LOCAL_DIRECTORY_NAME: Final[str] = (
    "CLOUD_FUNCTION_LOCAL_DIRECTORY_NAME=cloud_function"
)
_REQUIRED_USER_ENVIRONMENTAL_VARIABLES: Final[list] = [
    "CATEGORICAL_UNIQUE_VALUES_PATH",
    "COLUMN_METADATA_PATH",
    "DATA_BUCKET_NAME",
    "DEPLOYMENT_BUCKET_NAME",
    "HYPERPARAMETERS_OVERRIDES_PATH",
    "PROJECT_ID",
    "PROJECT_NUMBER",
    "REGION",
    "OUTPUT_CLASSES_PATH",
]
_DEPENDENT_ENVIRONMENTAL_VARIABLES: Final[dict] = dict(
    VERTEX_PIPELINE_ROOT="vertex_pipeline",
    CHECKPOINT_DIRECTORY_PATH="checkpoints",
    ARTIFACT_DIRECTORY_PATH="artifacts",
    TRAINING_LOGS_DIRECTORY_PATH="training_logs",
)
_DEPLOYMENT_BUCKET_NAME_KEY: Final[str] = "DEPLOYMENT_BUCKET_NAME"
_VERTEX_PIPELINE_PACKAGE_PATH_KEY: Final[str] = "VERTEX_PIPELINE_PACKAGE_PATH"
_REGION_KEY: Final[str] = "REGION"
_PROJECT_ID_KEY: Final[str] = "PROJECT_ID"
_ARTIFACT_REGISTRY_KEY: Final[str] = "ARTIFACT_REGISTRY"
_VERTEX_PIPELINE_DOCKER_IMAGE_NAME_KEY: Final[str] = (
    "VERTEX_PIPELINE_DOCKER_IMAGE_NAME"
)
_VERTEX_PIPELINE_LOCAL_DIRECTORY_NAME_KEY: Final[str] = (
    "VERTEX_PIPELINE_LOCAL_DIRECTORY_NAME"
)
_VERTEX_PIPELINE_PACKAGE_NAME: Final[str] = "optimus-pipeline.json"
_ENVIRONMETAL_VARIABLES_FILE_DESTINATION: Final[str] = "./optimus.env"


def assemble_independent_environmental_variables() -> str:
  return ("\n").join([
      _ARTIFACT_REGISTRY_NAME,
      _CLOUD_RUN_DOCKER_IMAGE_NAME,
      _CLOUD_RUN_APP_HOME_DIRECTORY,
      _VERTEX_PIPELINE_DOCKER_IMAGE_NAME,
      _VERTEX_PIPELINE_NAME,
      _CLOUD_RUN_SERVICE_NAME,
      _CLOUD_FUNCTION_NAME,
      _CLOUD_SOURCE_REPOSITORY_NAME,
      _VERTEX_PIPELINE_LOCAL_DIRECTORY_NAME,
      _CLOUD_FUNCTION_LOCAL_DIRECTORY_NAME,
      _CLOUD_BUILD_TRIGGER_NAME,
  ])


def load_user_environmental_variables() -> str:
  """Returns user environmental variables.

  Raises:
    ValueError: An error when user provided environmental variables miss any of
    the required values.
  """
  with tf.io.gfile.GFile(
      _USER_ENVIRONMENTAL_VARIABLES_PATH.value, "r"
  ) as artifact:
    user_environment_variables = artifact.read()
  processed_user_environment_variables = [
      variable.split("=")[0]
      for variable in user_environment_variables.split("\n")
      if variable
  ]
  processed_user_environment_variables.sort()
  missing_environmental_variables = list(
      set(_REQUIRED_USER_ENVIRONMENTAL_VARIABLES)
      - set(processed_user_environment_variables)
  )
  if missing_environmental_variables:
    raise ValueError(
        f"{missing_environmental_variables} environmental variables are"
        " missing."
    )
  return user_environment_variables


def extract_environmental_variables(
    *, user_environment_variables: str, independent_environmental_variables: str
) -> Mapping[str, str]:
  """Returns a mapping of environmental variables and their values required for later variable processing.

  Args:
    user_environment_variables: A string with user environmental variables.
    independent_environmental_variables: A string with independent environmental
      variables.
  """
  user_environment_variables_list = user_environment_variables.split("\n")
  deployment_bucket_name = [
      variable
      for variable in user_environment_variables_list
      if variable.startswith(_DEPLOYMENT_BUCKET_NAME_KEY)
  ][0].split("=")[1]
  region = [
      variable
      for variable in user_environment_variables_list
      if variable.startswith(_REGION_KEY)
  ][0].split("=")[1]
  project_id = [
      variable
      for variable in user_environment_variables_list
      if variable.startswith(_PROJECT_ID_KEY)
  ][0].split("=")[1]
  independent_environmental_variables_list = (
      independent_environmental_variables.split("\n")
  )
  artifact_registry = [
      variable
      for variable in independent_environmental_variables_list
      if variable.startswith(_ARTIFACT_REGISTRY_KEY)
  ][0].split("=")[1]
  vertex_pipeline_docker_image_name = [
      variable
      for variable in independent_environmental_variables_list
      if variable.startswith(_VERTEX_PIPELINE_DOCKER_IMAGE_NAME_KEY)
  ][0].split("=")[1]
  vertex_pipeline_local_directory_name = [
      variable
      for variable in independent_environmental_variables_list
      if variable.startswith(_VERTEX_PIPELINE_LOCAL_DIRECTORY_NAME_KEY)
  ][0].split("=")[1]
  return dict(
      deployment_bucket_name=deployment_bucket_name,
      region=region,
      project_id=project_id,
      artifact_registry=artifact_registry,
      vertex_pipeline_docker_image_name=vertex_pipeline_docker_image_name,
      vertex_pipeline_local_directory_name=vertex_pipeline_local_directory_name,
  )


def assemble_dependent_environment_variables(
    *, extracted_environmental_variables: Mapping[str, str]
) -> str:
  """Returns a string with dependent environmental variables.

  Args:
    extracted_environmental_variables: A mapping of environmental variables and
      their values required for later variable processing.
  """
  simple_dependent_variables = [
      key
      + "="
      + os.path.join(
          "gs://",
          extracted_environmental_variables["deployment_bucket_name"],
          value,
      )
      for key, value in _DEPENDENT_ENVIRONMENTAL_VARIABLES.items()
  ]
  vertex_pipeline_package_path = [
      _VERTEX_PIPELINE_PACKAGE_PATH_KEY
      + "="
      + os.path.join(
          "gs://",
          extracted_environmental_variables["deployment_bucket_name"],
          extracted_environmental_variables[
              "vertex_pipeline_local_directory_name"
          ],
          _VERTEX_PIPELINE_PACKAGE_NAME,
      )
  ]
  vertex_pipeline_docker_image = [
      _VERTEX_PIPELINE_DOCKER_IMAGE_NAME_KEY
      + "="
      + os.path.join(
          extracted_environmental_variables["region"] + "-docker.pkg.dev",
          extracted_environmental_variables["project_id"],
          extracted_environmental_variables["artifact_registry"],
          extracted_environmental_variables[
              "vertex_pipeline_docker_image_name"
          ],
      )
  ]
  return "\n".join(
      simple_dependent_variables
      + vertex_pipeline_package_path
      + vertex_pipeline_docker_image
  )


def assemble_environmental_variables(_) -> None:
  """Assembles environmental variables necessary for Optimus setup."""
  user_environment_variables = load_user_environmental_variables()
  independent_environmental_variables = (
      assemble_independent_environmental_variables()
  )
  extracted_environmental_variables = extract_environmental_variables(
      user_environment_variables=user_environment_variables,
      independent_environmental_variables=independent_environmental_variables,
  )
  dependent_environment_variables = assemble_dependent_environment_variables(
      extracted_environmental_variables=extracted_environmental_variables
  )
  all_environmental_variables = "\n".join([
      user_environment_variables,
      independent_environmental_variables,
      dependent_environment_variables,
  ])
  with tf.io.gfile.GFile(
      _ENVIRONMETAL_VARIABLES_FILE_DESTINATION, "w"
  ) as artifact:
    artifact.write(all_environmental_variables)


if __name__ == "__main__":
  flags.mark_flag_as_required("user_environmental_variables_path")
  app.run(assemble_environmental_variables)
