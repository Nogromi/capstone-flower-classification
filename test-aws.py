import requests

# aws lambda endpoint
url = 'https://xwy53vdg4i.execute-api.us-east-1.amazonaws.com/presenting/predict'

img_url = 'https://upload.wikimedia.org/wikipedia/commons/6/65/Ripe_fruits_by_Common_Dandelion.jpg'

data = {'url': img_url}
result = requests.post(url, json=data).json()
print(result)