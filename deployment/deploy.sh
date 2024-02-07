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

echo "Starting deploying Optimus on GCP."
echo "1/18 Installing Python packages required for deployment."
pip install --require-hashes -q --upgrade --force-reinstall -r requirements.txt --no-cache-dir
echo "2/18 Setting up environmental variables."
python -m setup.environmental_variables -user_environmental_variables_path=$USER_ENVIRONMETAL_VARIABLES_PATH
export $(grep -v "^#" "optimus.env" | xargs)
echo "3/18 Setting up GCP project."
gcloud auth application-default login
gcloud auth application-default set-quota-project $PROJECT_ID
gcloud auth login
echo "4/18 Enabling GCP APIs."
gcloud services enable compute.googleapis.com \
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
echo "5/18 Setting up Artifact Registry."
gcloud artifacts repositories create $ARTIFACT_REGISTRY_NAME --repository-format=docker --location=$REGION
yes | gcloud auth configure-docker $REGION-docker.pkg.dev
echo "6/18 Setting up GCS buckets."
gcloud storage buckets create gs://$DEPLOYMENT_BUCKET_NAME --location=$REGION
gcloud storage buckets create gs://$DATA_BUCKET_NAME --location=$REGION
gsutil iam ch serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com:roles/storage.objectAdmin gs://$DEPLOYMENT_BUCKET_NAME
gsutil iam ch serviceAccount:$PROJECT_NUMBER-compute@developer.gserviceaccount.com:roles/storage.objectAdmin gs://$DATA_BUCKET_NAME
echo "7/18 Setting up GCP permissions."
gcloud projects add-iam-policy-binding $PROJECT_ID --member=serviceAccount:service-$PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com --role=roles/aiplatform.user
gcloud projects add-iam-policy-binding $PROJECT_ID --member=serviceAccount:service-$PROJECT_NUMBER@gcp-sa-aiplatform-cc.iam.gserviceaccount.com --role=roles/artifactregistry.reader
echo "8/18 Creating metadata for the input data."
python -m setup.columns_metadata
echo "9/18 Creating training hyperparameters."
python -m setup.hyperparameters
echo "10/18 Creating required directories in the GCS buckets."
touch placeholder.txt
gsutil cp placeholder.txt gs://$DATA_BUCKET_NAME/training_experiences/placeholder.txt
gsutil cp placeholder.txt gs://$DEPLOYMENT_BUCKET_NAME/training_logs/placeholder.txt
rm placeholder.txt
echo "11/18 Creating and pushing a Docker image for a Cloud Run service."
cd $CLOUD_RUN_APP_HOME_DIRECTORY
docker build --tag=$REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_NAME/$CLOUD_RUN_DOCKER_IMAGE_NAME . \
--no-cache \
--build-arg CHECKPOINT_DIRECTORY_PATH=$CHECKPOINT_DIRECTORY_PATH \
--build-arg ARTIFACT_DIRECTORY_PATH=$ARTIFACT_DIRECTORY_PATH
cd ..
docker push $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_NAME/$CLOUD_RUN_DOCKER_IMAGE_NAME
echo "12/18 Creating a Cloud Run service."
gcloud run deploy $CLOUD_RUN_SERVICE_NAME --image $REGION-docker.pkg.dev/$PROJECT_ID/$ARTIFACT_REGISTRY_NAME/$CLOUD_RUN_DOCKER_IMAGE_NAME \
--max-instances=10 --min-instances=1 --port=80 \
--allow-unauthenticated --region $REGION \
--memory=2Gi --cpu=4 --concurrency=80 -q
echo "13/18 Creating a Pub/Sub topic triggering a Cloud Build service."
gsutil notification create -e OBJECT_FINALIZE -f json -p $(basename $TRAINING_LOGS_DIRECTORY_PATH)/ gs://$DEPLOYMENT_BUCKET_NAME
echo "14/18 Creating a Cloud Source Repository for a Cloud Build service."
gcloud source repos create $CLOUD_SOURCE_REPOSITORY_NAME
cd $CLOUD_RUN_APP_HOME_DIRECTORY
sudo rm -R artifacts checkpoints __pycache__ .ipynb_checkpoints ||:
git init
git config --global user.email "optimus@optimus.com"
git config --global user.name "optimus"
git config --global credential.https://source.developers.google.com.helper gcloud.sh
git remote add google https://source.developers.google.com/p/$PROJECT_ID/r/$CLOUD_SOURCE_REPOSITORY_NAME
git add * --force .dockerignore
git commit -m "Commiting files for a Cloud Run model server app."
git push --all google
cd ..
echo "15/18 Creating a Cloud Build service."
gcloud builds triggers create pubsub \
--name $CLOUD_BUILD_TRIGGER_NAME \
--topic projects/$PROJECT_ID/topics/$DEPLOYMENT_BUCKET_NAME \
--build-config cloudbuild.yaml \
--repo https://source.developers.google.com/p/$PROJECT_ID/r/$CLOUD_SOURCE_REPOSITORY_NAME \
--branch master \
--repo-type CLOUD_SOURCE_REPOSITORIES \
--region $REGION \
--service-account projects/$PROJECT_ID/serviceAccounts/$PROJECT_NUMBER-compute@developer.gserviceaccount.com \
--substitutions _REGION=$REGION,_PROJECT_ID=$PROJECT_ID,_ARTIFACT_REGISTRY_NAME=$ARTIFACT_REGISTRY_NAME,_CLOUD_RUN_DOCKER_IMAGE_NAME=$CLOUD_RUN_DOCKER_IMAGE_NAME,_CHECKPOINT_DIRECTORY_PATH=$CHECKPOINT_DIRECTORY_PATH,_ARTIFACT_DIRECTORY_PATH=$ARTIFACT_DIRECTORY_PATH,_CLOUD_RUN_SERVICE_NAME=$CLOUD_RUN_SERVICE_NAME,_DEPLOYMENT_BUCKET_NAME=$_DEPLOYMENT_BUCKET_NAME,_PROJECT_NUMBER=$PROJECT_NUMBER,_DEPLOYMENT_BUCKET_NAME=$DEPLOYMENT_BUCKET_NAME
echo "16/18 Creating and pushing a Docker image for a Vertex AI pipeline."
cd $VERTEX_PIPELINE_LOCAL_DIRECTORY_NAME
docker build --tag=$VERTEX_PIPELINE_DOCKER_IMAGE_NAME . --no-cache \
--build-arg VERTEX_PIPELINE_DOCKER_IMAGE_NAME=$VERTEX_PIPELINE_DOCKER_IMAGE_NAME
docker push $VERTEX_PIPELINE_DOCKER_IMAGE_NAME
cd ..
echo "17/18 Creating a Vertex AI Pipeline package."
python -m vertex_pipeline.vertex_pipeline
echo "18/18 Creating a Cloud Function triggering a Vertex AI Pipeline."
pip -q download -r $CLOUD_FUNCTION_LOCAL_DIRECTORY_NAME/requirements.txt --only-binary=:all: \
-d $CLOUD_FUNCTION_LOCAL_DIRECTORY_NAME/wheel_dependencies --python-version 3.10 \
--platform manylinux2014_x86_64 \
--implementation cp \
--no-cache-dir
cd $CLOUD_FUNCTION_LOCAL_DIRECTORY_NAME
gcloud functions deploy $CLOUD_FUNCTION_NAME --region $REGION \
--entry-point trigger_vertex_pipeline \
--stage-bucket $DEPLOYMENT_BUCKET_NAME \
--trigger-bucket $DATA_BUCKET_NAME \
--runtime python310 \
--source . \
--docker-registry=artifact-registry \
--update-build-env-vars GOOGLE_VENDOR_PIP_DEPENDENCIES=wheel_dependencies/ \
--run-service-account projects/$PROJECT_ID/serviceAccounts/$PROJECT_NUMBER-compute@developer.gserviceaccount.com \
--set-env-vars DATA_BUCKET_NAME=$DATA_BUCKET_NAME,ARTIFACT_DIRECTORY_PATH=$ARTIFACT_DIRECTORY_PATH,TRAINING_LOGS_DIRECTORY_PATH=$TRAINING_LOGS_DIRECTORY_PATH,PROJECT_ID=$PROJECT_ID,REGION=$REGION,VERTEX_PIPELINE_NAME=$VERTEX_PIPELINE_NAME,VERTEX_PIPELINE_PACKAGE_PATH=$VERTEX_PIPELINE_PACKAGE_PATH,VERTEX_PIPELINE_ROOT=$VERTEX_PIPELINE_ROOT
sudo rm -R wheel_dependencies
cd ..
sudo rm optimus.env
echo "Finished Optimus deployment."
