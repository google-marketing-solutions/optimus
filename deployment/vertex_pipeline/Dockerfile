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

FROM python:3.10

ARG VERTEX_PIPELINE_DOCKER_IMAGE_NAME

ENV VERTEX_PIPELINE_DOCKER_IMAGE_NAME $VERTEX_PIPELINE_DOCKER_IMAGE_NAME

COPY . .

RUN pip install --require-hashes -r requirements.txt
