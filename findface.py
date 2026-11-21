import cv2
import torch
import torch.nn.functional as f
import torchvision.transforms as transforms
from backbone import Backbone
from PIL import Image
import os
import glob
from config import (
    camera_index,
    frame_skip,
    detection_frame_size,
    model_det_path
)

class FaceDetector:
    def __init__(self):

        self.cap  = cv2.VideoCapture(camera_index)

        self.frame_skip = frame_skip
        self.frame_count = 0
        self.detection_frame_size = detection_frame_size

        self.detector = cv2.FaceDetectorYN.create(
            model=model_det_path,
            config="",
            input_size=detection_frame_size,
            score_threshold=0.5,
            nms_threshold=0.3,
            top_k=10
        )

    def run(self):

        while True:

            ret, frame = self.cap.read()

            if not ret:
                print("Не удалось получить кадр с камеры")
                break

            self.frame_count += 1
            if self.frame_count % self.frame_skip == 0:

                faces = self.detect_faces(frame)

                if faces:
                    self.send_notification(f"Обнаружено {len(faces)} лиц(а) на кадре")
                    print(f"Обнаружено {len(faces)} лиц(а) на кадре")
                else:
                    print("Лица не обнаружены")

                self.frame_count = 0

            cv2.imshow('Face Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27:
                break

        self.cap.release()

    def detect_faces(self, frame):

        resized_frame = cv2.resize(frame, self.detection_frame_size)

        faces = self.detector.detect(resized_frame)

        detected_faces = []
        if faces[1] is not None:
            for detection in faces[1]:

                x, y, w, h = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])

                x = int(x * frame.shape[1] / self.detection_frame_size[0])
                y = int(y * frame.shape[0] / self.detection_frame_size[1])
                w = int(w * frame.shape[1] / self.detection_frame_size[0])
                h = int(h * frame.shape[0] / self.detection_frame_size[1])

                face_rect = type('FaceRect', (), {
                    'left': lambda self, x=x: x,
                    'top': lambda self, y=y: y,
                    'right': lambda self, x=x, w=w: x + w,
                    'bottom': lambda self, y=y, h=h: y + h,
                    'width': lambda self, w=w: w,
                    'height': lambda self, h=h: h
                })()

                detected_faces.append(face_rect)

        return detected_faces

    def send_notification(self, message):

        print(message)