import pickle
from glob import glob
from tqdm import tqdm
import mediapipe as mp
import pandas as pd
from sklearn import preprocessing

label_encoder = preprocessing.LabelEncoder()

image_files = glob("dataset/*/*.*")

labels = pd.read_csv("dataset/labels.csv", index_col=0)

# encoding
labels["label"] = label_encoder.fit_transform(labels["label"])
labels["pth"] = labels["pth"].apply(lambda x: "dataset/" + x)

target_mapping = pd.Series(labels.label.values, index=labels.pth).to_dict()
# print(target_mapping)
# print(image_files)


def save_label_mapping():
    label_encoder_dict = dict(
        zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))
    )

    pickle.dump(label_encoder_dict, open("dataset/label_encoder_dict.pkl", "wb"))


BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

model_path = "model/face_landmarker.task"
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE,
)

cols = []

N_points = 478

for i in range(N_points):
    # single_cols = ["x_" + str(i), "y_" + str(i), "z_" + str(i)]
    single_cols = ["x_" + str(i), "y_" + str(i)]
    cols.extend(single_cols)

cols.append("label")

records = []
with FaceLandmarker.create_from_options(options) as landmarker:
    for image in tqdm(image_files[:]):
        mp_image = mp.Image.create_from_file(image)
        face_landmarker_result = landmarker.detect(mp_image)
        if len(face_landmarker_result.face_landmarks) == 0:
            continue
        values = []
        for point in face_landmarker_result.face_landmarks[0]:
            # values.extend([point.x, point.y, point.z])
            values.extend([point.x, point.y])

        label = target_mapping.get(image)
        values.append(label)

        my_dict = dict(zip(cols, values))

        records.append(my_dict)

df = pd.DataFrame.from_records(records)

df.to_parquet("data_xy.parquet")
