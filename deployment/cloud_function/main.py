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

"""Template for Cloud Function triggering a Vertex AI pipeline."""

import os
from google.cloud import aiplatform
from google.cloud import storage


def trigger_vertex_pipeline(*args, **kwargs) -> None:
  """Triggers a Vertex AI Pipeline run."""
  del args, kwargs
  client = storage.Client()
  blobs = list(
      client.list_blobs(
          os.getenv("DATA_BUCKET_NAME"), prefix="training_experiences"
      )
  )
  file_names = [item.name for item in blobs]
  file_names.sort(reverse=True)
  most_recent_experience_file = os.path.join(
      f"gs://{os.getenv('DATA_BUCKET_NAME')}", file_names[0]
  )
  parameter_values = dict(
      recent_experiences=most_recent_experience_file,
      artifact_directory=os.getenv("ARTIFACT_DIRECTORY_PATH"),
      training_logs_path=os.getenv("TRAINING_LOGS_DIRECTORY_PATH"),
  )
  aiplatform.init(
      project=os.getenv("PROJECT_ID"),
      location=os.getenv("REGION"),
  )
  job = aiplatform.PipelineJob(
      display_name=os.getenv("VERTEX_PIPELINE_NAME"),
      template_path=os.getenv("VERTEX_PIPELINE_PACKAGE_PATH"),
      pipeline_root=os.getenv("VERTEX_PIPELINE_ROOT"),
      enable_caching=False,
      parameter_values=parameter_values,
  )
  job.submit(
      service_account=(
          f"{os.getenv('PROJECT_NUMBER')}-compute@developer.gserviceaccount.com"
      )
  )
