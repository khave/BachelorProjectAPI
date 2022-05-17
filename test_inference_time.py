import numpy as np
import mlflow as mlflow
import tensorflow as tf
from tensorflow.keras import applications
import cv2
import os
import time
import requests

DATASET_FOLDER = 'Z:/SDU/6. Semester/Bachelor Projekt/Datasets/dataset 224x224 fully-built-big [ALL sets]/test'
MODEL_PATH = 'D:/Programmering/BachelorProjectAPI/models/combined_efficientnetv2s_data_aug_fine_tuned'

def load_img(img_path, preprocessing=None):
    img = cv2.imread(img_path) #7 | 1000x800p london bus.jpg
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = img.astype(np.float32)
    #img = img / 255.0
    img_array = np.expand_dims(img, axis=0)

    if preprocessing is not None:
        img_array = preprocessing(img_array)
    
    return img_array


def test_model(model, img_path, preprocessing=None):
    img_array = load_img(img_path, preprocessing)
    start = time.time()
    model.predict(img_array)
    end = time.time()
    # Get the time taken to make the prediction in seconds and milliseconds
    time_taken = end - start
    return time_taken


def test_api(num_images=1000):
    addr = 'http://127.0.0.1:5000'

    content_type = 'image/jpeg'
    headers = {'Content-Type': content_type}

    # Loop through all files in the dataset folder
    total_time = 0
    count = 0
    for root, dirs, files in os.walk(DATASET_FOLDER):
        for file in files:

            if count >= num_images:
                break

            start = time.time()
            img_path = path = os.path.join(root, file)
            img = cv2.imread(img_path)
            _, img_encoded = cv2.imencode('.jpg', img)

            response = requests.post(addr + '/predict', files={"img": open(img_path, 'rb')})
            end = time.time()
            time_taken = end - start

            print(f"Time taken: {time_taken}")
            total_time += time_taken
            count += 1

    print(f"Average time taken for API per prediction: {total_time / count} seconds")


def test_local_model():
    print("Loading model...")
    model = mlflow.keras.load_model(MODEL_PATH)
    print("Model loaded!")
    model_name = os.path.basename(MODEL_PATH)
    # Loop through all files in the dataset folder
    total_time = 0
    count = 0
    for root, dirs, files in os.walk(DATASET_FOLDER):
        for file in files:
            img_path = path = os.path.join(root, file)

            time_taken = test_model(model, img_path, preprocessing=applications.efficientnet_v2.preprocess_input)
            print(f"Time taken: {time_taken}")
            total_time += time_taken
            count += 1

    print(f"Average time taken for {model_name} per prediction: {total_time / count} seconds")


def main():
    num_images = 100
    print(f"Testing API on {num_images} images...")
    test_api(num_images=num_images)


if __name__ == "__main__":
    main()