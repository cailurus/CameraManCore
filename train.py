import pandas as pd
from tensorflow import keras
import pickle
import numpy as np

dataset_path = "data.parquet"
dataset_config_path = "dataset/label_encoder_dict.pkl"

data_config = pickle.load(open(dataset_config_path, "rb"))

data = pd.read_parquet(dataset_path)

data = data.loc[data["label"] >= 0]

data["label"] = data["label"].astype(np.int32)

val_dataframe = data.sample(frac=0.2, random_state=1337)
train_dataframe = data.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)

train_features = train_dataframe.drop("label", axis=1)
train_targets = train_dataframe["label"]

val_features = val_dataframe.drop("label", axis=1)
val_targets = val_dataframe["label"]


NUM_CLASSES = len(data_config.keys())

model = keras.Sequential(
    [
        keras.layers.Input((train_features.shape[-1],)),
        keras.layers.Dense(30, activation="elu"),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(20, activation="elu"),
        keras.layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

opt = keras.optimizers.Adam(learning_rate=0.01)
model.compile(
    optimizer=opt, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoints/model_at_epoch_{epoch}.h5", save_best_only=True
    )
]

model.fit(
    train_features,
    train_targets,
    batch_size=512,
    epochs=3000,
    verbose=2,
    callbacks=callbacks,
    validation_data=(val_features, val_targets),
)
