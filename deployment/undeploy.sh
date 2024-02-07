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

echo "Starting undeploying Optimus from GCP."
echo "1/11 Installing Python packages required for undeployment."
pip install -q --upgrade --force-reinstall -r requirements.txt --no-cache-dir
echo "2/11 Setting up environmental variables."
python -m setup.environmental_variables -user_environmental_variables_path=$USER_ENVIRONMETAL_VARIABLES_PATH
export $(grep -v "^#" "optimus.env" | xargs)
echo "3/11 Setting up GCP project."
gcloud auth application-default login
gcloud auth application-default set-quota-project $PROJECT_ID
gcloud auth login
echo "4/11 Removing Artifact Registry."
gcloud artifacts repositories delete $ARTIFACT_REGISTRY_NAME \
--location=$REGION --quiet
gcloud artifacts repositories delete gcf-artifacts \
--location=$REGION --quiet
echo "5/11 Removing GCS buckets."
gcloud storage rm --recursive gs://$DEPLOYMENT_BUCKET_NAME
gcloud storage rm --recursive gs://$DATA_BUCKET_NAME
gcloud storage rm --recursive gs://gcf-sources-$PROJECT_NUMBER-$REGION
echo "6/11 Removing a Cloud Run service."
gcloud run services delete $CLOUD_RUN_SERVICE_NAME --region $REGION --quiet
echo "7/11 Removing a Pub/Sub topic."
gcloud pubsub topics delete $DEPLOYMENT_BUCKET_NAME
echo "8/11 Removing Cloud Source Repository."
gcloud source repos delete $CLOUD_SOURCE_REPOSITORY_NAME --quiet
echo "9/11 Removing a Cloud Build service."
gcloud builds triggers delete $CLOUD_BUILD_TRIGGER_NAME --region $REGION
echo "10/11 Removing a Cloud Function."
gcloud functions delete $CLOUD_FUNCTION_NAME --region $REGION --quiet
echo "11/11 Disabling GCP APIs."
gcloud services disable compute.googleapis.com \
                       containerregistry.googleapis.com \
                       aiplatform.googleapis.com \
                       cloudbuild.googleapis.com \
                       cloudfunctions.googleapis.com \
                       artifactregistry.googleapis.com \
                       storage-component.googleapis.com \
                       run.googleapis.com \
                       apigateway.googleapis.com \
                       servicemanagement.googleapis.com \
                       servicecontrol.googleapis.com \
                       iam.googleapis.com \
                       pubsub.googleapis.com \
                       sourcerepo.googleapis.com
sudo rm optimus.env
echo "Undeployed Optimus from GCP project."
