import requests

BASE = "http://127.0.0.1:5000/"

# response = requests.get(BASE + "helloworld/dana")

# response = requests.post(BASE + "helloworld/dana", {"likes": 10, "views": 12, "name": "Video Name"})

# response = requests.put(BASE + "helloworld/dana", {"likes": 10, "views": 12, "name": "Video Name"})
# print(response.json())

# input()

# response = requests.get(BASE + "helloworld/danass")

# print(response.json())

# response = requests.delete(BASE + "helloworld/dana")
# print(response)

response = requests.post(BASE + "helloworld/dana", {"likes": 10, "views": 12, "name": "Video Name"})

print(response.json())
