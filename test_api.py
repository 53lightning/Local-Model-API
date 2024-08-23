import requests

# url for local api
url = "http://localhost:5000/inference"

# prompt for llm
prompt = {'text': 'What are some cool places to visit in Costa Rica?'}

# retrieving and printing the response
response = requests.post(url, json = prompt).json()['response']
print(response)