# %%
import mediapipe as mp
import cv2
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


model_path = './face_landmarker.task'
IMAGE_FILES = glob("dataset/*.png")

# %%
import mediapipe as mp

BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

feature_cols = []
for i in range(478):
    single_cols = ["x_"+str(i), "y_"+str(i), "z_"+str(i)]
    feature_cols.extend(single_cols)

feature_cols.append("label")

values = []
with FaceLandmarker.create_from_options(options) as landmarker:
    for image in IMAGE_FILES[:8]:
        mp_image = mp.Image.create_from_file(image)
        face_landmarker_result = landmarker.detect(mp_image)
        
        point_coord = []
        for point in face_landmarker_result.face_landmarks[0]:
            point_coord.extend([point.x, point.y, point.z])
    
        point_coord.append(1)
        my_dict = dict(zip(feature_cols, point_coord))
        values.append(my_dict)

df = pd.DataFrame.from_records(values)

df.to_parquet("data.parquet") 
# %%

data = pd.read_parquet("data.parquet")

val_df = data.sample(frac=0.2, random_state=1337)
train_df = data.drop(val_df.index)

def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("label")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds

train_ds = dataframe_to_dataset(train_df)
val_ds = dataframe_to_dataset(val_df)

for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)


train_ds = train_ds.batch(4)
val_ds = val_ds.batch(4)


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    #feature_ds = dataset.map(lambda x, y: x[name])
    #feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    #normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature

all_inputs = []
all_features_encoded = []
for feature in tqdm(feature_cols[:-1]):
    temp_feature = keras.Input(shape=(1,), name=feature)
    temp_feature_encoded = encode_numerical_feature(temp_feature, feature, train_ds)
    all_inputs.append(temp_feature)
    all_features_encoded.append(temp_feature_encoded)


all_features = layers.concatenate(
   all_features_encoded 
)

x = layers.Dense(478*3, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "binary_crossentropy", metrics=["accuracy"])
keras.utils.plot_model(model, show_shapes=True, rankdir="LR")
# %%
model.fit(train_ds, epochs=1, validation_data=val_ds)

# %%
for idx, file in enumerate(IMAGE_FILES):
    label_name = file.rsplit("/",1)[-1]
    label_name = label_name.rsplit("\\",1)[0]
    label = encode_label(label_name,category)
    image = cv2.imread(file)
    image = cv2.flip(image, 1)  # Mirror display
    debug_image = copy.deepcopy(image)

    # Detection implementation
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = face_mesh.process(image)
    image.flags.writeable = True
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:

            # Landmark calculation
            landmark_list = calc_landmark_list(debug_image, face_landmarks)

            # Conversion to relative coordinates / normalized coordinates
            pre_processed_landmark_list = pre_process_landmark(
                landmark_list)
            # Write to the dataset file
            logging_csv(label, pre_processed_landmark_list)