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
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Rescaling, RandomFlip, RandomRotation, RandomContrast
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, f1_score
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


def define_model_architecture(NUM_CLASSES, IMG_WIDTH, IMG_HEIGHT):
    model = Sequential([
    Rescaling(1./255), # Rescale input, so it doesn't have to be an extra step
    RandomFlip("horizontal"),
    RandomRotation(0.1),
        
    # Rest of model
    Conv2D(16, (3, 3), padding='same', activation='relu', input_shape=(IMG_WIDTH,IMG_HEIGHT, 3)),
    #Conv2D(32, (3, 3), kernel_regularizer=l2(0.001)),
    BatchNormalization(),
    Conv2D(16, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    #Dropout(0.25),

    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(32, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    #Dropout(0.25),

    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2,2)),
    #Dropout(0.25),

    #Conv2D(128, (3, 3), padding='same', activation='relu', kernel_regularizer=l2(0.001)),
    #Conv2D(128, (3, 3), activation='relu', kernel_regularizer=l2(0.001)),
    #BatchNormalization(),
    #MaxPooling2D(),
    #Dropout(0.25),

    Flatten(),
    Dense(128, activation='relu'),
    #Dropout(0.25),
    Dense(NUM_CLASSES, activation='softmax')
    ])

    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    
    return model

    

def load_dataset(IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE):
    DATASET_FOLDER = '/opt/ml/input/data/training'

    # Maybe batch size 16 or 32?. Test set was definitely 1 as evaluation shows.
    train_it = image_dataset_from_directory(f'{DATASET_FOLDER}/train/', validation_split=0.2, subset="training", label_mode='categorical', batch_size=BATCH_SIZE, shuffle=True, image_size=(IMG_WIDTH,IMG_HEIGHT), seed=42)
    val_it = image_dataset_from_directory(f'{DATASET_FOLDER}/train/', validation_split=0.2, subset="validation", label_mode='categorical', batch_size=BATCH_SIZE, shuffle=True, image_size=(IMG_WIDTH,IMG_HEIGHT), seed=42)
    test_it = image_dataset_from_directory(f'{DATASET_FOLDER}/test/', label_mode='categorical', batch_size=1, shuffle=True, image_size=(IMG_WIDTH,IMG_HEIGHT))

    AUTOTUNE = tf.data.AUTOTUNE
    train_it = train_it.cache().prefetch(buffer_size=AUTOTUNE)
    val_it = val_it.cache().prefetch(buffer_size=AUTOTUNE)
    test_it = test_it.cache().prefetch(buffer_size=AUTOTUNE)

    #datagen = ImageDataGenerator(validation_split=0.2, rescale=1.0/255.0) Don't need to use rescale now that we have rescaling as a layer
    #datagen = ImageDataGenerator(validation_split=0.2)
    #test_datagen = ImageDataGenerator()
    #train_it = datagen.flow_from_directory(f'{DATASET_FOLDER}/train/', subset="training", class_mode='categorical', batch_size=BATCH_SIZE, target_size=(IMG_WIDTH,IMG_HEIGHT), shuffle=True, seed=42)
    #val_it = datagen.flow_from_directory(f'{DATASET_FOLDER}/train/', subset="validation", class_mode='categorical', batch_size=BATCH_SIZE, target_size=(IMG_WIDTH,IMG_HEIGHT), shuffle=True, seed=42)
    #test_it = test_datagen.flow_from_directory(f'{DATASET_FOLDER}/test/', class_mode='categorical', batch_size=1, target_size=(IMG_WIDTH,IMG_HEIGHT), shuffle=True, seed=42)

    
    # .class_names for image_dataset_from_directory and .labels for flow_from_directory
    #class_names = train_it.class_names
    #LOGGER.info(f"class names: {class_names}")
    #print(f"class names: {class_names}")

    return train_it, val_it, test_it


def test_model(model, test_it):
    # Test scores
    score = model.evaluate(test_it, verbose=0)
    mlflow.log_metric("test_loss", score[0])
    mlflow.log_metric("test_accuracy", score[1])    
    
    # Confusion Matrix
    labels = ['21034', '31058', '31088', '31112', '31113', '31114', '31121', '40220', '40468', '40469', '40532', '42134', '75280', '75297', '75299', '76901', '76902']
    
    pred = model.predict(test_it)


    # Convert predictions classes to one hot vectors 
    Y_pred_classes = np.argmax(pred, axis=1) 
    # Convert validation observations to one hot vectors
    y_test = np.concatenate([y for x, y in test_it], axis=0)
    Y_true = np.argmax(y_test, axis=1)


    print(Y_pred_classes.shape)
    print(Y_true.shape)

    X_test = np.concatenate([x for x, y in test_it], axis=0)


    # Errors are difference between predicted labels and true labels
    errors = (Y_pred_classes - Y_true != 0)

    Y_pred_classes_errors = Y_pred_classes[errors]
    Y_pred_errors = pred[errors]
    Y_true_errors = Y_true[errors]
    X_test_errors = X_test[errors]

    cm = confusion_matrix(Y_true, Y_pred_classes) 
    thresh = cm.max() / 2.

    fig, ax = plt.subplots(figsize=(12,12))
    im, cbar = heatmap(cm, labels, labels, ax=ax,
                   cmap=plt.cm.Blues, cbarlabel="count of predictions")
    texts = annotate_heatmap(im, data=cm, threshold=thresh)

    fig.tight_layout()
    # Use mlflow to log the figure
    plt.savefig('confusion_matrix.png')
    mlflow.log_artifact('confusion_matrix.png')
    plt.show()
    plt.close()

    # Get the precision, recall, f1-score, and support for each class and log them to mlflow
    #precision = precision_score(Y_true, Y_pred_classes, average=None)
    #recall = recall_score(Y_true, Y_pred_classes, average=None)
    #f1 = f1_score(Y_true, Y_pred_classes, average=None)
    #support = recall_score(Y_true, Y_pred_classes, average=None)

    #for i in range(len(labels)):
    #    mlflow.log_metric(f"precision_{labels[i]}", precision[i])
    #    mlflow.log_metric(f"recall_{labels[i]}", recall[i])
    #    mlflow.log_metric(f"f1_{labels[i]}", f1[i])
    #    mlflow.log_metric(f"support_{labels[i]}", support[i])    
    clf_report = classification_report(Y_true, Y_pred_classes, target_names=labels, output_dict=True)
    sns_classification_report = sns.heatmap(pd.DataFrame(clf_report).iloc[:-1, :].T, annot=True)
    sns_classification_report.figure.savefig('classification_report.png')
    mlflow.log_artifact('classification_report.png')



def heatmap(data, row_labels, col_labels, ax=None, cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.
    """
    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)
    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)
    
    ax.set_xlabel('Predicted Label') 
    ax.set_ylabel('True Label')
    
    return im, cbar


def annotate_heatmap(im, data=None, fmt="d", threshold=None):
    """
    A function to annotate a heatmap.
    """
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = im.axes.text(j, i, format(data[i, j], fmt), horizontalalignment="center",
                                 color="white" if data[i, j] > threshold else "black")
            texts.append(text)

    return texts


def preprocess_dataset(train_it, val_it, test_it):
    # Normalize the images 
    train_it = train_it.map(lambda x, y: (tf.image.per_image_standardization(x), y))
    val_it = val_it.map(lambda x, y: (tf.image.per_image_standardization(x), y))
    test_it = test_it.map(lambda x, y: (tf.image.per_image_standardization(x), y))
    return train_it, val_it, test_it


def augment_dataset(dataset):
    data_augmentation = Sequential(
        [
            RandomFlip("horizontal"),
            RandomRotation(0.1),
        ]
    )
    augmented_dataset = dataset.map(lambda x, y: (data_augmentation(x, training=True), y))
    return augmented_dataset


if __name__ == "__main__":

    #training_configuration = load_hyperparameter_config_file()

    # get the model & setup mlflow
    mlflow.set_tracking_uri("model_store_url")
    mlflow.set_experiment("kristian_lego_classification_test")
    mlflow.sklearn.autolog()
    
    # Global variables
    RUN_NAME = "CNN-Combined-Data-Aug"
    IMG_WIDTH = 224
    IMG_HEIGHT = 224
    BATCH_SIZE = 32 
    NUM_CLASSES = 17
    EPOCHS=1000 # Usually write 1000, but with early stopping with for example 8 epochs with restore best weight
    # Also do reduce lr on plateu, so it can more optimally learn. 
    # Look at the curve

    train_it, val_it, test_it = load_dataset(IMG_WIDTH, IMG_HEIGHT, BATCH_SIZE)

    # Augmenting training set
    #train_it = augment_dataset(train_it)

    
    model = define_model_architecture(NUM_CLASSES, IMG_WIDTH, IMG_HEIGHT)
    
    # get epochs from config file
    with mlflow.start_run(run_name=RUN_NAME):
        mlflow.keras.autolog()
        
        history = model.fit(
        train_it,
        validation_data=val_it,
        epochs=EPOCHS,
        #steps_per_epoch=train_it.__len__() // (BATCH_SIZE*2), # Reduce steps per epoch to train on a finer piece of the dataset
        workers=4,
        verbose=2,
        callbacks=[callbacks.EarlyStopping(patience=7, monitor='val_loss', restore_best_weights=True),
                   callbacks.ReduceLROnPlateau(patience=3, monitor='val_loss', factor=0.1, mode='auto')]
        )
        # Use callback to reduce LR on plateu of validation loss. Then maybe start with a higher learning rate.
        test_model(model, test_it)
        
        # Save model summary as mlflow artifact
        with open('model_summary.txt', 'w') as f:
            model.summary(print_fn=lambda x: f.write(x + '\n'))
        mlflow.log_artifact('model_summary.txt')

    # Perform test evaluation

    # get current model's metrics on test set
    #calculate_test_metrics(
    #    model=model,
    #    x_test=np.array(split_dataset["x_test"]),
    #    y_test=np.array(split_dataset["y_test"]),
    #    labels=training_configuration["classes"],
    #    run_name=f"original_{training_configuration['model_name']}",
    #)

    LOGGER.info(
        "Training complete!"
    )
    print("Training complete")
