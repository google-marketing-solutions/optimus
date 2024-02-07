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

"""Template for a Optimus Cloud Run model server."""

import contextlib
import datetime
import json
import logging
import os
import pickle
from typing import Any, List, Mapping, Union
import fastapi
from fastapi.openapi import utils
from ml_collections.config_dict import config_dict
from optimus.agent_lib import agents
from optimus.preprocessing_lib import base_preprocessing
from optimus.trainer_lib import base_trainer
import pandas as pd
import pydantic
import tensorflow as tf


_BUILDING_BLOCKS = {}


@contextlib.asynccontextmanager


async def lifespan(app: fastapi.FastAPI) -> None:


  """Yields the model server before its ready to make predictions.

  It reduces the time it take for the model to then make live predictions.

  Args:
    app: The model server application.
  """
  if not os.getenv("CLOUD_RUN_APP_HOME_DIRECTORY"):
    logging.info(
        "`CLOUD_RUN_APP_HOME_DIRECTORY` environmental variable must be set."
    )
  _BUILDING_BLOCKS["home_directory"] = os.getenv("CLOUD_RUN_APP_HOME_DIRECTORY")
  with tf.io.gfile.GFile(
      os.path.join(
          _BUILDING_BLOCKS["home_directory"],
          "artifacts",
          "hyperparameters.json",
      ),
      "rb",
  ) as artifact:
    hyperparameters = config_dict.ConfigDict(json.load(artifact))
  _BUILDING_BLOCKS["agent"] = agents.get_agent()(
      hyperparameters=hyperparameters
  )
  _BUILDING_BLOCKS["model_state"] = (
      base_trainer.initialize_model_state_for_prediction(
          agent=_BUILDING_BLOCKS["agent"],
          hyperparameters=hyperparameters,
      )
  )
  with tf.io.gfile.GFile(
      os.path.join(
          _BUILDING_BLOCKS["home_directory"],
          "artifacts",
          "column_metadata.pickle",
      ),
      "rb",
  ) as artifact:
    column_metadata = config_dict.ConfigDict(pickle.load(artifact))
  categorical_columns_unique_values_path = os.path.join(
      _BUILDING_BLOCKS["home_directory"],
      "artifacts",
      "categorical_unique_values.pickle",
  )
  categorical_columns_encoding_mapping_path = os.path.join(
      _BUILDING_BLOCKS["home_directory"],
      "artifacts",
      "categorical_values_encoding.pickle",
  )
  output_classes_encoding_path = os.path.join(
      _BUILDING_BLOCKS["home_directory"],
      "artifacts",
      "output_classes_encoding.pickle",
  )
  optimus_data_preprocessor = base_preprocessing.BaseDataPreprocessor(
      columns=column_metadata.all_columns,
      skip_columns=column_metadata.skipped_columns,
      categorical_columns=column_metadata.categorical_columns,
      categorical_columns_unique_values_path=categorical_columns_unique_values_path,
      categorical_columns_encoding_mapping_path=categorical_columns_encoding_mapping_path,
      output_classes_encoding_path=output_classes_encoding_path,
      action_space=hyperparameters.action_space,
  )
  # Preloading the key class attrbutes
  _ = optimus_data_preprocessor._skip_columns_indexes
  _ = optimus_data_preprocessor.categorical_columns
  _ = optimus_data_preprocessor.categorical_columns_indexes
  _ = optimus_data_preprocessor.categories_mappings
  _ = optimus_data_preprocessor._unknown_categorical_encoding_value
  _ = optimus_data_preprocessor.output_classes_encoding
  _BUILDING_BLOCKS["preprocessor"] = optimus_data_preprocessor
  yield
  _BUILDING_BLOCKS.clear()


app = fastapi.FastAPI(lifespan=lifespan)


class RequestItem(pydantic.BaseModel):
  """A class specifying the request input type.

  Attributes:
    instances: The compatible input type from an API request.
  """

  instances: List[Mapping[str, Any]]


@app.post("/predictions", response_model=None)
def model_serve(
    request: RequestItem,
) -> Mapping[str, Union[int, List[float], List[int], List[bool]]]:
  """Returns predictions with the most optimal actions.

  Args:
      request: A list with mappings between feature names and their values.
  """
  user_context = _BUILDING_BLOCKS["preprocessor"].preprocess_data(
      input_data=pd.DataFrame.from_records(request.instances).to_numpy(),
  )
  prediction = _BUILDING_BLOCKS["agent"].predict(
      agent_state=_BUILDING_BLOCKS["model_state"],
      batch=user_context,
      prediction_seed=datetime.datetime.now().microsecond,
  )
  postprocessed_action = _BUILDING_BLOCKS["preprocessor"].postprocess_data(
      input_data=prediction.action
  )
  return dict(
      prediction_seed=prediction.prediction_seed,
      state=prediction.state.tolist(),
      action=prediction.action.tolist(),
      postprocessed_action=postprocessed_action,
      value=prediction.value.tolist(),
      log_probability=prediction.log_probability.tolist(),
      done=prediction.done.tolist(),
      attentive_transformer_loss=prediction.attentive_transformer_loss.tolist(),
  )


@app.get("/api/healthz")
def health_check() -> Mapping[str, bool]:
  """Returns a mapping with a confirmation the model server runs properly."""
  return dict(api_healthy=True)


def custom_openapi() -> app.openapi_schema:
  """Returns the schema for the application."""
  if app.openapi_schema:
    return app.openapi_schema
  openapi_schema = utils.get_openapi(
      title="FastAPI", version="0.0.1", routes=app.routes
  )
  openapi_schema["x-google-backend"] = dict(
      address="${CLOUD_RUN_URL}",
      deadline="${TIMEOUT}",
  )
  openapi_schema["paths"]["/predictions"]["options"] = dict(
      operationId="corsHelloWorld",
      responses={"200": dict(desciption="Successful response")},
  )
  app.openapi_schema = openapi_schema
  return app.openapi_schema
