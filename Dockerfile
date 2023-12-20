FROM public.ecr.aws/lambda/python:3.10
RUN pip install pipenv

COPY ["Pipfile","Pipfile.lock", "./"]
RUN pip install pipenv
RUN pipenv install --system --deploy

COPY models/flower-model.tflite models/

COPY lambda_function.py .

CMD [ "lambda_function.lambda_handler" ]