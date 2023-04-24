import requests

response = requests.post('https://urlhaus-api.abuse.ch/v1/predict')

print(response.status_code)
print(response.json())