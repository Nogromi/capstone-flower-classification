# This is the repo of the capstone project for the 2023 [machine learning zoomcamp](https://github.com/alexeygrigorev/mlbookcamp-code/tree/master/course-zoomcamp) 


## Table of contents

- [1. About the project](#topic1)
- [2. Project files and folder explained](#topic2)
- [3. Dataset](#topic3)
- [4. Running this project](#topic4)
- [4. Setup project locally](#topic5)
- [5.  Project deployment](#topic6)

<h2 id="topic1"> 1. About the project</h2> 

This capstone project is a part of the [Machine Learning Zoomcamp course](https://github.com/DataTalksClub/machine-learning-zoomcamp) held by [DataTalks.Club](https://datatalks.club/) aims to create a  **Real-Time Flowers Classification Service**. It utilize finetuned CNN that was containerized via docker and pushed to AWS ECR where, finally the image is used to build AWS Lambda Function 

The model learns from the data to distinguish the species. When provided with new image of flower, the model will be able to generate probabilities of every class. Following  this training, the model can be deployed for real-world usage in a production environment.

<img src="media/christmas_flowers_dataset_0.jpg" alt=Flwers width="600" height="400">



<h2 id="topic2"> 2. Project important files and folders explained</h2> 

- **flowers/\***  The dataset contains 16 sub-directories, one per class: There are 15,740 total images:

- **EDA.ipynb:** This is a Jupyter Notebook where I carried out exploratory (EDA) data analysis on the dataset.
- **train.ipynb:** This is a Jupyter Notebook where I carried out trainng phase (Using  **Saturn Cloud**).

- **models/xception_v2_01_0.811.h5:** trained model
- **models/flower-model.tflite:** smaller tflite model

- **train.py:** This is the Python script used to train the model
- **test.py:** This is the Python script used to test the model locally
- **test-aws.py:** This is the Python script used to test aws lambda function. 
- **convert.py:** This is the Python script used to convert keras model to regular tflite model
  
- **lambda_function.py** This is the main Python script. Contains  preppocessing, and  uses tflite_runtime model for inference. it wil be the function on AWS Lamba

- **Pipfile & Pipfile.lock** These are configuration files used for managing Python dependencies and packages in this project

- **dockerfile** This is a docker image file for deploying the model to docker container

Everything is put into contaner on AWS ECR and an AWS Lambda Function is created from the ECR image.  

<h2 id="topic3"> 3. Dataset</h2> 

To download the dataset into root folder use this [link](https://www.kaggle.com/datasets/l3llff/flowers/). You can utilize handy lib `opendatasets` for that. Get  `kaggle.json` from  https://www.kaggle.com/settings and put the file into the root folder (or insert login, api_key in the input). Run first cell in `train.ipynb`.

<h2 id="topic4"> 4. Running this project</h2> 
<h3 id="running 1">To try out this project, follow  these steps:</h3> 

1. Clone this repository by running this command: `git@github.com:Nogromi/capstone1.git`
2. Enter the project directory:` cd capstone1`. If you dont want to set project locally or create container just skip to step 6
 
<h3 id="running 3"> Running docker Container</h3> 

3. Build the docker image: docker `docker build -t flower:v1 .`
4. Run the docker container: `dockerdocker run  -it --rm -p 8080:8080 flower:v1` (On windows, make sure you have docker running.) 
5. In another terminal and run `cd capstone1`. Next, run `python test.py`. This will prompt you to enter a URL sample, so enter any URL sample to get a prediction of whether the URL is malicious or good.

<h3 id="running 4">Fetch AWS Lambda Function</h3> 

6. Alternativelly, run `python test-aws.py` 
This script request AWS Lambda Function. I will keep this  service running approzimatelly for two weeks after 12/18/2023 project.

<h2  id="topic5">5. Setup project locally</h3> 

<h3> Setting up the environment</h3> 

```bash
pip install pipenv
pipenv install --dev # this will install both development and production dependencies
pipenv shell #Activate virtual environment
```
<h3> Running Notebooks</h3> 


   
<h2 id="topic6"> 6. Project deployment</h2> 

If you have any questions, reach out to me via email at anatolii.krivko@gmail.com
