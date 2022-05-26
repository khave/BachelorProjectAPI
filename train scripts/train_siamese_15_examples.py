import argparse
import json
import logging
import os
from collections import defaultdict
from typing import List, Optional

import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Rescaling, Layer
from tensorflow.keras import callbacks, optimizers, Model
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
from tensorflow.keras import applications, metrics
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
#from tensorflow.keras.layers.experimental.preprocessing import Rescaling
# Since we're using Tensorflow 2.1.3, we cannot have Rescaling as a layer



LOGGER = logging.getLogger()
LOGGER.setLevel(logging.INFO)

# Changing hardware configs
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
config = tf.compat.v1.ConfigProto(
    gpu_options=tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.9)
)
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

os.environ["MLFLOW_S3_UPLOAD_EXTRA_ARGS"] = json.dumps(
    {"ACL": "bucket-owner-full-control"}
)


def _parse_args():
    """
    Parses SageMaker Pipeline estimator call inputs and parameters, as specified when invoking estimator.fit().
    :return parsed Arguments
    """

    parser = argparse.ArgumentParser()

    # Data, model, and output directories
    # model_dir is always passed in from SageMaker. By default this is a S3 path under the default bucket.
    parser.add_argument(
        "--model_dir", type=str, help="The directory where the model will be stored."
    )
    parser.add_argument(
        "--train",
        type=str,
        default=os.environ.get("SM_CHANNEL_TRAINING"),
        help="The directory where the dataset is stored.",
    )

    return parser.parse_known_args()


# Provided two tensors t1 and t2
# Euclidean distance = sqrt(sum(square(t1-t2)))
def euclidean_distance(vects):
    """Find the Euclidean distance between two vectors.

    Arguments:
        vects: List containing two tensors of same length.

    Returns:
        Tensor containing euclidean distance
        (as floating point value) between vectors.
    """

    x, y = vects
    sum_square = tf.math.reduce_sum(tf.math.square(x - y), axis=1, keepdims=True)
    return tf.math.sqrt(tf.math.maximum(sum_square, tf.keras.backend.epsilon()))


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


def define_model_architecture(IMG_WIDTH, IMG_HEIGHT):
    # Loading efficientnet model
    embedding = mlflow.keras.load_model('s3://lego-bds-modelstore-mlflow-default/modelstore/32/ee8ce7629136407eb51a240377142831/artifacts/model')  
    embedding.trainable = False
    
    x = embedding.layers[-1].output
    x = tf.keras.layers.BatchNormalization(name="unique_batchnorm")(x)
    #x = Dense(512, activation='relu', name='unique_dense_1')(x)
    #x = Dense(256, activation='relu', name='unique_dense_2')(x)
    #x = Dense(128, activation='relu', name='unique_dense_3')(x)
    prediction = Dense(128, activation="relu", name='feature_extraction_output')(x)
    embedding_network = tf.keras.Model(embedding.input, prediction)
    
    #embedding_network = Sequential(
    #    [
    #        x,
    #        tf.keras.layers.BatchNormalization(),
    #        Dense(10, activation="tanh")
    #    ]
    #)
    
    
    
    #input = layers.Input((IMG_WIDTH, IMG_HEIGHT, 3))
    #x = tf.keras.layers.BatchNormalization()(input)
    #x = layers.Conv2D(4, (3, 3), activation="tanh")(x)
    #x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    #x = layers.Conv2D(16, (3, 3), activation="tanh")(x)
    #x = layers.AveragePooling2D(pool_size=(2, 2))(x)
    #x = layers.Flatten()(x)

    #x = tf.keras.layers.BatchNormalization()(x)
    #x = layers.Dense(10, activation="tanh")(x)
    #embedding_network = tf.keras.Model(input, x)


    input_1 = layers.Input((IMG_WIDTH, IMG_HEIGHT, 3))
    input_2 = layers.Input((IMG_WIDTH, IMG_HEIGHT, 3))

    # As mentioned above, Siamese Network share weights between
    # tower networks (sister networks). To allow this, we will use
    # same embedding network for both tower networks.
    tower_1 = embedding_network(input_1)
    tower_2 = embedding_network(input_2)

    merge_layer = layers.Lambda(euclidean_distance)([tower_1, tower_2])
    normal_layer = tf.keras.layers.BatchNormalization()(merge_layer)
    
    x = Dense(128, activation='relu')(normal_layer)
    #norm2 = tf.keras.layers.BatchNormalization()(fc1)
    x = Dense(64, activation='relu')(x)
    x = Dense(32, activation='relu')(x)
    
    output_layer = layers.Dense(1, activation="sigmoid")(x)
    siamese = tf.keras.Model(inputs=[input_1, input_2], outputs=output_layer)
    
    return siamese


def make_pairs(x, y):
    """Creates a tuple containing image pairs with corresponding label.

    Arguments:
        x: List containing images, each index in this list corresponds to one image.
        y: List containing labels, each label with datatype of `int`.

    Returns:
        Tuple containing two numpy arrays as (pairs_of_samples, labels),
        where pairs_of_samples' shape is (2len(x), 2,n_features_dims) and
        labels are a binary array of shape (2len(x)).
    """

    num_classes = max(y) + 1
    digit_indices = [np.where(y == i)[0] for i in range(num_classes)]


    pairs = []
    labels = []

    for idx1 in range(len(x)):
        # add a matching example
        x1 = x[idx1]
        label1 = y[idx1]
        idx2 = np.random.choice(digit_indices[label1])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [1]

        # add a non-matching example
        label2 = np.random.randint(0, num_classes - 1)
        while label2 == label1:
            label2 = np.random.randint(0, num_classes - 1)

        idx2 = np.random.choice(digit_indices[label2])
        x2 = x[idx2]

        pairs += [[x1, x2]]
        labels += [0]

    return np.array(pairs), np.array(labels).astype("float32")


def load_dataset(NUM_IMAGES=1000, BATCH_SIZE=1):
    DATASET_FOLDER = '/opt/ml/input/data/training'

    train_it = image_dataset_from_directory(f'{DATASET_FOLDER}/train/', label_mode='int', batch_size=BATCH_SIZE, shuffle=True, image_size=(IMG_WIDTH,IMG_HEIGHT), seed=42)
    #val_it = image_dataset_from_directory(f'{DATASET_FOLDER}/train/', validation_split=0.2, subset="validation", label_mode='int', batch_size=BATCH_SIZE, shuffle=True, image_size=(IMG_WIDTH,IMG_HEIGHT), seed=42)
    #test_it = image_dataset_from_directory(f'{DATASET_FOLDER}/test/', label_mode='int', batch_size=1, shuffle=True, image_size=(IMG_WIDTH,IMG_HEIGHT), seed=42)
    
    
    # Preporcess according to our resnet. Don't preprocess when working with the CNN
    train_it = train_it.map(lambda x, y: (applications.efficientnet_v2.preprocess_input(x), y))
    #val_it = val_it.map(lambda x, y: (applications.efficientnet_v2.preprocess_input(x), y))
    #test_it = test_it.map(lambda x, y: (applications.efficientnet_v2.preprocess_input(x), y))

    AUTOTUNE = tf.data.AUTOTUNE
    train_it = train_it.cache().prefetch(buffer_size=AUTOTUNE)
    #val_it = val_it.cache().prefetch(buffer_size=AUTOTUNE)
    #test_it = test_it.cache().prefetch(buffer_size=AUTOTUNE)

    #train_it = train_it.take(NUM_IMAGES)
    # Take 20% of 1000
    #val_it = val_it.take(int(NUM_IMAGES*0.2)) # 20% for validation
    #test_it = test_it.take(int(NUM_IMAGES*0.2)) # 20% for test

    x_train = []
    y_train = []
    #x_val = []
    #y_val = []
    #x_test = []
    #y_test = []

    for image, label in train_it:
        x_train.append(image[0])
        y_train.append(label[0])

    #for image, label in val_it:
    #    x_val.append(image[0])
    #    y_val.append(label[0])
    
    #for image, label in test_it:
    #    x_test.append(image[0])
    #    y_test.append(label[0])

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    #x_val = np.array(x_val)
    #y_val = np.array(y_val)
    #x_test = np.array(x_test)
    #y_test = np.array(y_test)

    x_train = x_train.astype(np.float32)
    #x_val = x_val.astype(np.float32)
    #x_test = x_test.astype(np.float32)

    return x_train, y_train


def test_model(model, x_test_1, x_test_2, labels_test):
    score = model.evaluate([x_test_1, x_test_2], labels_test)
    mlflow.log_metric("test_loss", score[0])
    mlflow.log_metric("test_accuracy", score[1])    


if __name__ == "__main__":

    #training_configuration = load_hyperparameter_config_file()

    # get the model & setup mlflow
    mlflow.set_tracking_uri("model_store_url")
    mlflow.set_experiment("kristian_lego_classification_test")
    mlflow.sklearn.autolog()
    
    # Global variables
    RUN_NAME = "Siamese-ENet-Fully-15-Plus-5-Sets"
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    BATCH_SIZE = 1 
    NUM_CLASSES = 17
    MARGIN = 1
    EPOCHS=10 # Usually write 1000, but with early stopping with for example 8 epochs with restore best weight
    # Also do reduce lr on plateu, so it can more optimally learn. 
    # Look at the curve

    x_train, y_train = load_dataset(NUM_IMAGES=3000, BATCH_SIZE=BATCH_SIZE)

    # We have 1000, 200, 200 split because of 20-20 split
    pairs_train, labels_train = make_pairs(x_train, y_train) # 1000
    #pairs_val, labels_val = make_pairs(x_val, y_val) # 200
    #pairs_test, labels_test = make_pairs(x_test, y_test) # 200


    # Split the training pairs
    x_train_1 = pairs_train[:, 0]
    x_train_2 = pairs_train[:, 1]

    # Split the validation pairs
    #x_val_1 = pairs_val[:, 0]
    #x_val_2 = pairs_val[:, 1]

    # Split the test pairs
    #x_test_1 = pairs_test[:, 0]
    #x_test_2 = pairs_test[:, 1]


    # Presproccess data according to transfer learned architecture
    #train_it = train_it.map(lambda x, y: (applications.resnet_v2.preprocess_input(x), y))
    #val_it = val_it.map(lambda x, y: (applications.resnet_v2.preprocess_input(x), y))
    #test_it = test_it.map(lambda x, y: (applications.resnet_v2.preprocess_input(x), y))
    
    #model = define_model_architecture(NUM_CLASSES, IMG_WIDTH, IMG_HEIGHT)
    

    #base_cnn = tf.keras.applications.resnet_v2.ResNet50V2(
    #weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False
    #)

    #flatten = Flatten()(base_cnn.output)
    #dense1 = Dense(512, activation="relu")(flatten)
    #dense1 = BatchNormalization()(dense1)
    #dense2 = Dense(256, activation="relu")(dense1)
    #dense2 = BatchNormalization()(dense2)
    #output = Dense(256)(dense2)

    #embedding = Model(base_cnn.input, output, name="Embedding")

    siamese_model = define_model_architecture(IMG_WIDTH, IMG_HEIGHT)

    # Compile the model
    siamese_model.compile(loss=loss(margin=MARGIN), optimizer=optimizers.Adam(learning_rate=0.0001), metrics=["accuracy"])
    
    # get epochs from config file
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.keras.autolog()

        history = siamese_model.fit(
            [x_train_1, x_train_2],
            labels_train,
            #validation_data=([x_val_1, x_val_2], labels_val),
            epochs=EPOCHS,
            #steps_per_epoch=train_it.__len__() // (BATCH_SIZE*2), # Reduce steps per epoch to train on a finer piece of the dataset
            workers=4,
            verbose=2,
            callbacks=[callbacks.EarlyStopping(patience=3, monitor='loss', restore_best_weights=True),
                    callbacks.ReduceLROnPlateau(monitor='loss', factor=0.1, patience=2)],
        )
        
        #test_model(siamese_model, x_test_1, x_test_2, labels_test)

        # Save model summary as mlflow artifact
        with open('model_summary.txt', 'w') as f:
            siamese_model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact('model_summary.txt')

        # Use callback to reduce LR on plateu of validation loss. Then maybe start with a higher learning rate.
        #test_model(siamese_model, test_it)

    LOGGER.info(
        "Training complete!"
    )
    print("Training complete")
