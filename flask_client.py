import requests
import json
import cv2
import base64

addr = 'http://localhost:5000'

content_type = 'image/jpeg'
headers = {'Content-Type': content_type}

img_path = 'Z:/SDU/6. Semester/Bachelor Projekt/Datasets/dataset 224x224 partially-built-big [ALL sets]/test/21034/frame_0.jpg'
img = cv2.imread(img_path)
_, img_encoded = cv2.imencode('.jpg', img)

response = requests.post(addr + '/predict', files={"img": open(img_path, 'rb')})

print(json.loads(response.text))