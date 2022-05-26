# BachelorProjectAPI
API to predict LEGO sets based on images

Use 
```
curl --location --request POST 'http://localhost:5000/predict' \
--form 'img=@"<img_path>"'
```

to predict an image or use the Postman collection as provided.

You can also go to the Swagger documentation to test it. 
http://localhost:5000/apidocs/

## Files Explanation
**Siamese_Test.ipynb**:  
Contains the code used to test the Siamese network in the form of a Jupyter Notebook

**data_pipeline.py**:  
Contains the code used to generate the dataset from a series of videos and a folder structure where each class is a folder.

**flask_api.py**:  
Contains the Flask API for the models

**flask_api_siamese.py**:  
Contains the Flask API for the Siamese models 

**flask_client.py**:  
Contains code for sending an image to the Flask API

**test_inference_time.py**:  
Contains the code to test the API response times

**train scripts folder**:  
Contains all the scripts used for training the model