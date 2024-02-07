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

"""Template for a Vertex AI pipeline."""

import os
import tempfile

from absl import app
from kfp import compiler
from kfp import dsl
import tensorflow as tf


@dsl.component(
    base_image=os.getenv("VERTEX_PIPELINE_DOCKER_IMAGE_NAME"),
)
def get_artifacts(
    *,
    artifact_directory: str,
    hyperparameters: dsl.Output[dsl.Artifact],
) -> None:
  """Loads all the artifacts required for model retraining.

  Args:
    artifact_directory: A path to a directory where the artifacts are stored.
  """



  import json
  import os
  import tensorflow as tf
  from ml_collections.config_dict import config_dict




  hyperparameters_path = os.path.join(
      artifact_directory, "hyperparameters.json"
  )
  if tf.io.gfile.exists(hyperparameters_path):
    with tf.io.gfile.GFile(hyperparameters_path, "rb") as artifact:
      experiment_hyperparameters = config_dict.ConfigDict(json.load(artifact))
  else:
    raise ValueError(
        f"There is no hyperparameters path: {hyperparameters_path}."
    )
  with tf.io.gfile.GFile(hyperparameters.path + ".json", "w") as artifact:
    json.dump(experiment_hyperparameters.to_dict(), artifact)


@dsl.component(
    base_image=os.getenv("VERTEX_PIPELINE_DOCKER_IMAGE_NAME"),
)
def get_data(
    *,
    recent_experiences: str,
    ingested_recent_experiences: dsl.Output[dsl.Dataset],
) -> None:
  """Loads all the data required for model retraining.

  Args:
    recent_experiences: A path to a CSV file containing the most recent
      experiences used to train the model.
  """



  import pandas as pd




  recent_experiences = pd.read_csv(recent_experiences, low_memory=False)
  recent_experiences.to_csv(
      ingested_recent_experiences.path + ".csv", index=False
  )


@dsl.component(
    base_image=os.getenv("VERTEX_PIPELINE_DOCKER_IMAGE_NAME"),
)
def train_optimus(
    *,
    training_logs_path: str,
    ingested_recent_experiences: dsl.Input[dsl.Dataset],
    hyperparameters: dsl.Input[dsl.Artifact],
) -> None:
  """Retrains the Optimus model and saves a file with confirmation on completion.

  Args:
    training_logs_path: A path to a directory where training confirmation logs
      should be saved.
  """



  import os
  import datetime
  from custom_code import custom_reward
  import json
  import tensorflow as tf
  from optimus.agent_lib import agents
  from optimus.trainer_lib import trainers
  from optimus.data_pipeline_lib import data_pipelines
  import pandas as pd
  from ml_collections.config_dict import config_dict
  import numpy as np




  with tf.io.gfile.GFile(hyperparameters.path + ".json", "r") as artifact:
    experiment_hyperparameters = config_dict.ConfigDict(json.load(artifact))

  recent_experiences = pd.read_csv(
      ingested_recent_experiences.path + ".csv", low_memory=False
  ).values
  data_pipeline = data_pipelines.get_data_pipeline(
      data_pipeline_name=experiment_hyperparameters.data_pipeline_name
  )(hyperparameters=experiment_hyperparameters)
  reward = custom_reward.CustomReward(
      hyperparameters=experiment_hyperparameters,
  )
  train_pipeline_iterator = iter(
      data_pipeline.train_data_pipeline(
          train_data=recent_experiences.astype(np.float32),
          reward_calculation_function=reward.calculate_reward,
      )
  )
  agent = agents.get_agent()(hyperparameters=experiment_hyperparameters)
  trainer = trainers.get_trainer()(
      agent=agent,
      reward=reward,
      hyperparameters=experiment_hyperparameters,
  )
  for experience_batch in train_pipeline_iterator:
    trainer.train(train_data=experience_batch)
  if not tf.io.gfile.exists(training_logs_path):
    tf.io.gfile.makedirs(training_logs_path)
  logs_path = os.path.join(
      training_logs_path,
      "completion_log_"
      + datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
      + ".txt",
  )
  with tf.io.gfile.GFile(logs_path, "w") as training_log:
    training_log.write("Training_complete.")


@dsl.pipeline(
    pipeline_root=os.getenv("PIPELINE_ROOT"),
    name=os.getenv("PIPELINE_NAME"),
)
def run_optimus_pipeline(
    recent_experiences: str,
    artifact_directory: str,
    training_logs_path: str,
) -> None:
  """Provides a description of an end-to-end model retraining pipeline.

  Args:
    recent_experiences: A path to a CSV file containing the most recent
      experiences used to train the model.
    artifact_directory: A path to a directory where the artifacts are stored.
    training_logs_path: A path to a directory where training confirmation logs
      should be saved.
  """
  data_operation = get_data(recent_experiences=recent_experiences)
  artifacts_operation = get_artifacts(artifact_directory=artifact_directory)
  training_operation = train_optimus(
      ingested_recent_experiences=data_operation.outputs[
          "ingested_recent_experiences"
      ],
      hyperparameters=artifacts_operation.outputs["hyperparameters"],
      training_logs_path=training_logs_path,
  )


def compile_and_upload_pipeline_package(_) -> None:
  """Compiles and uploads the Vertex AI Pipeline package to a GCS bucket."""
  with tempfile.TemporaryDirectory() as temporary_directory:
    pipeline_package_path = os.path.join(
        temporary_directory, "optimus-pipeline.json"
    )
    compiler.Compiler().compile(
        pipeline_func=run_optimus_pipeline,
        package_path=pipeline_package_path,
    )
    tf.io.gfile.copy(
        pipeline_package_path,
        os.getenv("VERTEX_PIPELINE_PACKAGE_PATH"),
        overwrite=True,
    )


if __name__ == "__main__":
  app.run(compile_and_upload_pipeline_package)
