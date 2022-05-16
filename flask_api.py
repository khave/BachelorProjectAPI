from flask import Flask, request
import numpy as np
import mlflow as mlflow
import cv2
import base64
import json

app = Flask(__name__)

print("Loading model...")
model_path = 'Models/EfficientNetV2S/EfficientNetV2S_Data_Aug_Fine_Tuned'
model = mlflow.keras.load_model(model_path)
print("Model loaded!")

labels = ['21034', '31058', '31088', '31112', '31113', '31114', '31121', '40220', '40468', '40469', '40532', '42134', '75280', '75297', '75299', '76901', '76902']


class NumpyFloatValuesEncoder(json.JSONEncoder):
    '''
        Class to encode numpy float32 values to json
    '''
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)


@app.route('/predict', methods=['POST'])
def predict():
    print("Making prediction...")
    data = request.files["img"]
    img = cv2.imdecode(np.fromstring(data.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)
    pred = model.predict(img)
    
    # Format the prediction as a JSON object with the label and probability of the predicted classes
    pred_dict = {}
    for i in range(len(labels)):
        pred_dict[labels[i]] = pred[0][i]
    
    print(f"Prediction made and got label: {labels[np.argmax(pred)]}")
    return json.dumps(pred_dict, cls=NumpyFloatValuesEncoder)



if __name__ == '__main__':
    app.run(host='localhost', port=5000)
    print("Server running on port 5000")