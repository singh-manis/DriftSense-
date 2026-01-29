import requests

url = 'http://127.0.0.1:5000/upload'
files = {'file': open('test_log.csv', 'rb')}

try:
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print("Response Body:")
    print(response.text)
except Exception as e:
    print(f"Request failed: {e}")
