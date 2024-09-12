import requests

# url for local api
url = "http://localhost:5000/inference"

# prompt for llm
prompt = {'text': 'What are some cool places to visit in Costa Rica?',
          'max_length': 1024,
          'temperature': 0.7}

# retrieving and printing the response
response = requests.post(url, json = prompt).json()['response']
print(response)