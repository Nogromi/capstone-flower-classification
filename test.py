import requests

# call the docker url with the image url localhost:8080

# local test
url = 'http://localhost:8080/2015-03-31/functions/function/invocations'

# aws lambda endpoint
# url = 'https://xwy53vdg4i.execute-api.us-east-1.amazonaws.com/presenting/predict'

img_url = 'https://upload.wikimedia.org/wikipedia/commons/6/65/Ripe_fruits_by_Common_Dandelion.jpg'

data = {'url': img_url}
result = requests.post(url, json=data).json()
print(result)