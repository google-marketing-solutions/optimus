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

# 1. Builder container copies files from GCS
# 2. Serving container copies files from the builder container and other

# 1. MODEL FILE CONTAINER
FROM google/cloud-sdk:alpine AS builder

ARG CHECKPOINT_DIRECTORY_PATH

ENV CHECKPOINT_DIRECTORY_PATH $CHECKPOINT_DIRECTORY_PATH

ARG ARTIFACT_DIRECTORY_PATH

ENV ARTIFACT_DIRECTORY_PATH $ARTIFACT_DIRECTORY_PATH

# We're finding and copying only the latest checkpoint
RUN mkdir checkpoints && gsutil -m cp -r $(gsutil ls -d -l $CHECKPOINT_DIRECTORY_PATH/checkpoint* | sort -k 2 | tail -n 2 | head -1 | cut -c 34- | rev | cut -c2- | rev) checkpoints ||:

RUN mkdir artifacts && gsutil -m cp -r $ARTIFACT_DIRECTORY_PATH/* artifacts

# 2. SERVING CONTAINER
FROM tiangolo/uvicorn-gunicorn-fastapi:python3.10

# Enabling Knative logs
ENV PYTHONUNBUFFERED True

# Changing the root directory on the container
ENV CLOUD_RUN_APP_HOME_DIRECTORY "/cloud_run"

WORKDIR $CLOUD_RUN_APP_HOME_DIRECTORY

COPY requirements.txt $CLOUD_RUN_APP_HOME_DIRECTORY/

# Installing production dependencies
RUN pip install --require-hashes -r $CLOUD_RUN_APP_HOME_DIRECTORY/requirements.txt

# Copying the rest of the project files.
COPY . $CLOUD_RUN_APP_HOME_DIRECTORY

COPY --from=builder checkpoints ./checkpoints

COPY --from=builder artifacts ./artifacts

ENV PORT 80

ENV HOST "0.0.0.0"

CMD uvicorn main:app --host $HOST --port $PORT --loop asyncio
