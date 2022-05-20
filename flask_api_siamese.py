from flask import Flask, request
import numpy as np
import mlflow as mlflow
import cv2
import base64
import json
import os
import tensorflow as tf
from tensorflow.keras import applications


# Define constrastive loss
def loss(margin=1):
    """Provides 'constrastive_loss' an enclosing scope with variable 'margin'.

  Arguments:
      margin: Integer, defines the baseline for distance for which pairs
              should be classified as dissimilar. - (default is 1).

  Returns:
      'constrastive_loss' function with data ('margin') attached.
  """

    # Contrastive loss = mean( (1-true_value) * square(prediction) +
    #                         true_value * square( max(margin-prediction, 0) ))
    def contrastive_loss(y_true, y_pred):
        """Calculates the constrastive loss.

      Arguments:
          y_true: List of labels, each label is of type float32.
          y_pred: List of predictions of same length as of y_true,
                  each label is of type float32.

      Returns:
          A tensor containing constrastive loss as floating point value.
      """

        square_pred = tf.math.square(y_pred)
        margin_square = tf.math.square(tf.math.maximum(margin - (y_pred), 0))
        return tf.math.reduce_mean(
            (1 - y_true) * square_pred + (y_true) * margin_square
        )

    return contrastive_loss


app = Flask(__name__)

print("Loading model...")
model_path = 'models/siamese_enet_fully_15_bonsai'
model = mlflow.keras.load_model(model_path, custom_objects={"contrastive_loss": loss(margin=1)})
print("Model loaded!")

class NumpyFloatValuesEncoder(json.JSONEncoder):
    '''
        Class to encode numpy float32 values to json
    '''
    def default(self, obj):
        if isinstance(obj, np.float32):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def load_img(img_path):
    #TF version
    #img = tf.io.read_file(img_path)
    #img = tf.image.decode_png(img, channels=3)
    #img = tf.image.convert_image_dtype(img, tf.float32)
    #img = tf.image.resize(img, size=(224, 224))

    #CV version
    img = cv2.imread(img_path)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # Preprocess the input according to the model requirements:  efficientnet_v2 | resnet_v2
    img = applications.efficientnet_v2.preprocess_input(img)
    return img


def predict_k_way(input_image, k=4):
    # Load all images from the folder
    references_folder = "siamese_references/train"

    
    highest_mean = 0
    prediction_class = ""

    for root, dirs, files in os.walk(references_folder):
        # Get the folder name from root
        folder_name = root.split("\\")[-1]
        print(f"Testing: {folder_name}")
        # Define the support set that we will predict the images on
        pairs = []
        for file in files:
            # for each file
            img_path = os.path.join(root, file)
            supp_img = load_img(img_path)
            pairs += [[input_image[0], supp_img[0]]]

            # If the support set has reached k, then do predictions
            if len(pairs) >= k:
                pairs = np.array(pairs)
                # Split the pairs
                x_pair_1 = pairs[:, 0]
                x_pair_2 = pairs[:, 1]
                predictions = model.predict([np.array(x_pair_1), np.array(x_pair_2)])
                # Take the average of the predictions
                #print(predictions)
                mean_pred = np.mean(predictions, axis=0)
                print(f"Mean of predictions: {mean_pred}")
                if mean_pred > highest_mean:
                    highest_mean = mean_pred
                    prediction_class = folder_name
                break

    print(f"Predicted class: {prediction_class} with a mean similarity of {highest_mean} over {k} images")

    # Format the response to be sent back to the client
    response = {
        f"{prediction_class}": highest_mean[0]
    }
    return response


@app.route('/predict', methods=['POST'])
def predict():
    data = request.files["img"]
    img = cv2.imdecode(np.fromstring(data.read(), np.uint8), cv2.IMREAD_UNCHANGED)
    img = cv2.resize(img, (224, 224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = img.astype(np.float32)
    img = np.expand_dims(img, axis=0)

    # Preprocess the input according to the model requirements:  efficientnet_v2 | resnet_v2
    img = applications.efficientnet_v2.preprocess_input(img)

    response = predict_k_way(img, k=8)

    return json.dumps(response, cls=NumpyFloatValuesEncoder)


if __name__ == '__main__':
    print("Server running on port 5000")
    app.run(host='localhost', port=5000) # threaded=True 
    print("Goodbye!")