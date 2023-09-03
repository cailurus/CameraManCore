import mediapipe as mp
import tensorflow as tf
import numpy as np

BaseOptions = mp.tasks.BaseOptions
VisionRunningMode = mp.tasks.vision.RunningMode
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
FaceLandmarkerResult = mp.tasks.vision.FaceLandmarkerResult


class DetectionBase:
    def __init__(self):
        self.result = None

    def parse_result(self, result, output_image, timestamp_ms):
        self.result = result

    def inference(self, image, timestamp):
        raise NotImplementedError


class FaceModel(DetectionBase):
    def __init__(self):
        self.result = None
        self.setup()

    def setup(self):
        base_options = BaseOptions(model_asset_path="model/face_landmarker.task")
        options = FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
            running_mode=VisionRunningMode.LIVE_STREAM,
            num_faces=1,
            result_callback=self.parse_result,
        )
        self.model = FaceLandmarker.create_from_options(options)

    def inference(self, image, timestamp):
        self.model.detect_async(image, timestamp)
        return self.result


class EmotionModel:
    def __init__(self):
        self.interpreter = tf.lite.Interpreter(
            model_path="model/face_keypoint_classifier.tflite"
        )
        self.interpreter.allocate_tensors()

        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

    def inference(self, landmark_list):
        self.interpreter.set_tensor(
            self.input_details[0]["index"], np.array([landmark_list], dtype=np.float32)
        )

        self.interpreter.invoke()
        tflite_results = self.interpreter.get_tensor(self.output_details[0]["index"])

        inference_res = np.argmax(np.squeeze(tflite_results))

        return inference_res
