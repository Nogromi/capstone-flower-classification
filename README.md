# Capstone Project for the 2023 Machine Learning Zoomcamp

## Table of Contents

- [1. About the Project](#topic1)
- [2. Project Files and Folder Structure](#topic2)
- [3. Dataset](#topic3)
- [4. Running this Project](#topic4)
- [5. Set Up Project Locally](#topic5)
- [6. Project Deployment](#topic6)

<h2 id="topic1">1. About the Project</h2> 

This capstone project is part of the [Machine Learning Zoomcamp course](https://github.com/DataTalksClub/machine-learning-zoomcamp) held by [DataTalks.Club](https://datatalks.club/). Its goal is to create a **Real-Time Flowers Classification Service**. In this work, I designed three different models: Simple Perceptron, Vanilla CNN, and a finetuned Xception model (which was used as the final model). I applied techniques such as regularization, augmentation, and dependency removal of the TensorFlow library. Furthermore, this model and other dependencies were containerized and pushed to AWS ECR. Finally, that image was used to build an AWS Lambda Function.

### The Goal of the Project
The model learns from the data to distinguish the species. When provided with a new image of a flower, the model will be able to generate probabilities for every class. Following this training, the model can be deployed for real-world usage in a production environment.

![Flowers](media/christmas_flowers_dataset_0.jpg)

<h2 id="topic2">2. Project Files and Folder Structure</h2> 

This project contains the following files:

- **flowers/\***: The dataset contains 16 sub-directories, one per class, with a total of 15,740 images. (**Note:** this folder is not present in the repo, so you need to download [it](#topic3)).

- **EDA.ipynb:** This Jupyter Notebook carries out exploratory data analysis (EDA) on the dataset.
- **train.ipynb:** This Jupyter Notebook carries out the training phase (using [**Saturn Cloud**](https://app.community.saturnenterprise.io/)).

- **models/xception_v2_01_0.811.h5:** Trained model.
- **models/flower-model.tflite:** Converted TFLite model.

- **train.py:** The Python script used to train the model.
- **test.py:** The Python script used to test the model locally.
- **test-aws.py:** The Python script used to test the AWS Lambda function.
- **convert.py:** The Python script used to convert the Keras model to a TFLite model.
  
- **lambda_function.py:** This is the main Python file used in the deployment. It contains preprocessing and uses the TFLite runtime model for inference. It will be the function on AWS Lambda.

- **Pipfile & Pipfile.lock:** These are configuration files used for managing Python dependencies and packages in this project.

- **dockerfile:** This is a Docker image file for deploying the model to a Docker container.

Everything is put into a container on AWS ECR, and then an AWS Lambda Function is created from this ECR image.

<h2 id="topic3">3. Dataset</h2> 

To download the dataset into the root folder, use this [link](https://www.kaggle.com/datasets/l3llff/flowers/). You can utilize the handy library `opendatasets` for that. Get `kaggle.json` from [here](https://www.kaggle.com/settings) and put the file into the root folder (or insert login, api_key in the input). Run the first cell in `train.ipynb`.

<h2 id="topic4">4. Running this Project</h2> 
<h3 id="running1">To try out this project, follow these steps:</h3> 

1. Clone this repository by running this command: 

```bash
git clone git@github.com:Nogromi/capstone1.git
```
2. Enter the project directory:
```bash
cd capstone1
```

**FAST START HERE**: If you dont want to set project locally or create container just do steps 1, 2, 6.
 
<h3 id="running3"> Running docker Container</h3> 

3. Build the docker image: docker 
```bash
docker build -t flower:v1 .
```
4. Run the docker container(make sure your docker is running)^

```bash
docker run  -it --rm -p 8080:8080 flower:v1` 
```

5. In another terminal and run `cd capstone1`. Next, run

```bash
python test.py
```
 This will send some URL sample, to get a probability of flower species.

<h3 id="running4">Fetch AWS Lambda Function</h3> 

6. Alternativelly, run `python test-aws.py` 
This script requests AWS Lambda Function with some image url  in the script. Alternatively, you can replace with any other flower image url. I will keep this service running approzimatelly for two weeks after 12/18/2023 project.

<h2  id="topic5">5. Setup project locally</h3> 

<h3> Setting up the environment</h3> 

```bash
pip install pipenv
pipenv install --dev # this will install both development and production dependencies
pipenv shell #Activate virtual environment
```
<h3> Notebooks and code</h3> 

### Eda 

Run `eda.ipynb` to expore the dataset

### Training the model 
Run `train.ipynb` to experiment with three models. Perceptorn, CNN form scratch and finetuned Xception. Also you can explore optimization techniques such as regularisation, augmentation. Notebook was converted to a python file.

 Run the training script
```
python train.py (GPU required)
```
Script retrain the model with the dataset and then convert model  to `{model_name}.tflite` 

### Inference 
To chech model performance  and see how wee model predicts species run `inference.ipynb`
   
### lambda_function.py 
Explore this function to see how dependency remowal technique was applied. This script uses tflite_runtime library to run model in predict mode only. 

<h2 id="topic6"> 6. Project deployment</h2> 
 Use the container you created in section

  [Running docker Container](#running3)
  
  Create AWS ECR using command line
  Install AWS CLI utility: 
```bash
(base) $~> pip install awscli
```
note that I use different environment though which I access the root folder because I don't want install awscli to the pipenv

Configure your awscli profile. Reed more [here](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html). Alternativelly use any **how to** tutorial 
```bash
(base) $~> aws configure
...
```

Create ECR [Elastic Container Registry]:
```bash
aws ecr create-repository --repository-name flower-tflite
...
```
you will get the responce like this^
```bash

{
    "repository": {
        "repositoryArn": "arn:aws:ecr:us-east-1:387369431815:repository/flower-tflite",
        "registryId": "387369431815",
        "repositoryName": "flower-tflite",
        "repositoryUri": "387369431815.dkr.ecr.us-east-1.amazonaws.com/flower-tflite",
        "createdAt": 1702899561.0,
        "imageTagMutability": "MUTABLE",
        "imageScanningConfiguration": {
            "scanOnPush": false
        },
        "encryptionConfiguration": {
            "encryptionType": "AES256"
        }
    }
}
```
Now I will create some helpfull file to utilize.
Lets create new file where we will construct ECR REMOTE_URI:
1) save `repositoryUri` from the response above

```bash
"387369431815.dkr.ecr.us-east-1.amazonaws.com/flower-tflite" # repositoryUri
```
2) split it into such chunks:
```bash
ACCOUNT=387369431815
REGION=us-east-1
REGISTRY=flower-tflite
PREFIX=${ACCOUNT}.dkr.ecr.${REGION}.amazonaws.com/${REGISTRY}
TAG=flower-01
REMOTE_URI=${PREFIX}:${TAG}

TAG=facial-emotion-recognition-001
REMOTE_URI=${PREFIX}:${TAG}
```
Login to aws ecr
```bash
(base) $~> $(aws ecr get-login --no-include-email)
WARNING! Using --password via the CLI is insecure. Use --password-stdin.
Login Succeeded
```
copy and execute the lines from helpfull file in command line 
Check if `{REMOTE_URI}` is correct 
```bash
(base) $~> echo ${REMOTE_URI}
```
Tag the your local docker image  with the ${REMOTE_URI}
```bash
docker tag flower:v1 ${REMOTE_URI}
```

Finally

push image to ${REMOTE_URI} 

```bash
(base) $~> docker push ${REMOTE_URI}
```

# Create AWS Lambda Function

To create, configure, and test the Lambda function, follow the instructions in this [video](https://youtu.be/kBch5oD5BkY?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&t=457). Test it on the above URL and get identical results.

# Create and Configure AWS API Gateway

Follow the instructions [here](https://youtu.be/wyZ9aqQOXvs?list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR):

- Create a REST API (POST METHOD: /predict) with the Lambda function created above.
- Test the API Gateway using the `test-aws.py` script.

If you have any questions, reach out to me via email at anatolii.krivko@gmail.com
