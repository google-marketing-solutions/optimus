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

"""Support module to process hyperparameters."""

import os
from absl import app
from optimus import training_setup
from optimus.actions_lib import actions
from optimus.agent_lib import agents
from optimus.data_pipeline_lib import data_pipelines
from optimus.reward_lib import rewards
from optimus.trainer_lib import trainers


def assemble_and_save_hyperparameters(_) -> None:
  """Assembles and saves all experiment hyperparametrs to a GCS storage bucket."""
  actions_hyperparameters = actions.get_actions_hyperparameters()
  agent_hyperparameters = agents.get_agent_hyperparameters()
  reward_hyperparameters = rewards.get_reward_hyperparameters()
  data_pipeline_hyperparameters = (
      data_pipelines.get_data_pipeline_hyperparameters()
  )
  trainer_hyperparameters = trainers.get_trainer_hyperparameters()
  training_setup.set_hyperparameters(
      actions_hyperparameters=actions_hyperparameters,
      agent_hyperparameters=agent_hyperparameters,
      reward_hyperparameters=reward_hyperparameters,
      data_pipeline_hyperparameters=data_pipeline_hyperparameters,
      trainer_hyperparameters=trainer_hyperparameters,
      checkpoint_directory=os.getenv("CHECKPOINT_DIRECTORY_PATH"),
      artifact_directory=os.getenv("ARTIFACT_DIRECTORY_PATH"),
      hyperparameters_file=os.getenv("HYPERPARAMETERS_OVERRIDES_PATH"),
  )


if __name__ == "__main__":
  app.run(assemble_and_save_hyperparameters)
