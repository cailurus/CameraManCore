import csv
import copy
import itertools
import cv2 as cv
import mediapipe as mp
import numpy as np
import tensorflow as tf
from model import FaceModel, EmotionModel


def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n silence mode
        mode = 0
    if key == 107:  # k  record mode
        mode = 1
    if key == 32:  # space inference mode
        mode = 2
    return number, mode


def calc_landmark_list(face_landmarker_result):
    point_coord = []
    for point in face_landmarker_result.face_landmarks[0]:
        point_coord.extend([point.x, point.y])
    return point_coord


def logging_csv(number, mode, landmark_list):
    if mode == 0:
        pass
    if mode == 1 and (0 <= number <= 9):
        csv_path = "model/face_keypoints.csv"
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([number, *landmark_list])
    return


mode = 0
timestamp = 0
face_model = FaceModel()
emotion_model = EmotionModel()

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv.CAP_PROP_FPS, 24)

while True:
    # Process Key (ESC: end)
    key = cv.waitKey(10)
    if key == 27:  # ESC
        break

    class_label, mode = select_mode(key, mode)
    print(f"class_label: {class_label}, mode: {mode}")

    # Camera capture
    ret, frame = cap.read()
    timestamp += 1
    if not ret:
        break
    frame = cv.flip(frame, 1)  # Mirror display

    # Detection implementation
    # frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    keypoint_results = face_model.inference(mp_image, timestamp)

    if keypoint_results is not None:
        landmark_list = calc_landmark_list(keypoint_results)
        logging_csv(class_label, mode, landmark_list)

    else:
        print("No face detected")

    if mode == 1:
        cv.putText(
            frame,
            f"MODE: Record keypoints mode, {class_label}",
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

    if mode == 2:
        emotion_result = emotion_model.inference(landmark_list)
        cv.putText(
            frame,
            f"MODE: Inference mode, {emotion_result}",
            (10, 90),
            cv.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )

    cv.imshow("Facial Emotion Recognition", frame)

cap.release()
cv.destroyAllWindows()
