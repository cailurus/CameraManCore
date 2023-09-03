import cv2
import numpy as np
import pandas as pd
from glob import glob
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm
import keras
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Normalization

from sklearn.model_selection import train_test_split

dataset_path = "model/face_keypoints.csv"
model_save_path = "model/face_keypoint_classifier.hdf5"
tflite_save_path = "model/face_keypoint_classifier.tflite"


def build_dataset():
    X_dataset = np.loadtxt(
        dataset_path,
        delimiter=",",
        dtype="float32",
        usecols=list(range(1, (478 * 2) + 1)),
    )
    y_dataset = np.loadtxt(dataset_path, delimiter=",", dtype="int32", usecols=(0))
    X_train, X_val, y_train, y_val = train_test_split(
        X_dataset, y_dataset, train_size=0.8, random_state=32
    )

    return X_train, X_val, y_train, y_val


def build_model(num_classes):
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Input((478 * 2,)),
            tf.keras.layers.Dense(40, activation="elu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(20, activation="elu"),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(10, activation="elu"),
            tf.keras.layers.Dense(num_classes, activation="softmax"),
        ]
    )

    opt = keras.optimizers.Adam(learning_rate=0.0005)
    model.compile(
        optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )

    return model


X_train, X_val, y_train, y_val = build_dataset()

NUM_CLASSES = 3

print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
model = build_model(NUM_CLASSES)
model.fit(
    X_train,
    y_train,
    batch_size=128,
    epochs=3000,
    verbose=2,
    callbacks=[
        keras.callbacks.ModelCheckpoint(
            filepath="checkpoints/model_at_epoch_{epoch}.h5", save_best_only=True
        )
    ],
    validation_data=(X_val, y_val),
)

print("Save model")
model.save(model_save_path, include_optimizer=False)

print("Convert to tflite")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_quantized_model = converter.convert()
open(tflite_save_path, "wb").write(tflite_quantized_model)

print("Load tflite model and inference")
interpreter = tf.lite.Interpreter(model_path=tflite_save_path)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]["index"], np.array([X_val[0]]))
interpreter.invoke()
tflite_results = interpreter.get_tensor(output_details[0]["index"])

print(np.argmax(np.squeeze(tflite_results)))
