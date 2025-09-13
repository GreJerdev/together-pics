from typing import Tuple, Union, List
import mediapipe as mp

import cv2
from PIL import Image, ImageTk
from pathlib import Path
import numpy as np


class FaceFinder:

    def __init__(self, face_detection_model=None):

        if face_detection_model is None:
            self.mp_face_detection = mp.solutions.face_detection
            self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        else:
            self.face_detection = face_detection_model

    def detect_faces(self, image: cv2.typing.MatLike) -> Tuple[List[Tuple[int, int, int, int]], List[np.ndarray]]:

        if image is None:
            return [], []
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        face_locations = []
        face_crops = []

        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                face_locations.append((y, x + width, y + height, x))
                face_crop = rgb_image[y:y+height, x:x+width]
                face_crops.append(face_crop)
        
        return face_locations, face_crops


if __name__ == "__main__":
    face_finder = FaceFinder()
    image = cv2.imread("images (1).jpg")
    face_locations, face_crops = face_finder.detect_faces(image)
    print(face_locations)
    print(face_crops)
    cv2.imshow("Face Finder", image)
    for i in range(len(face_crops)):
        cv2.imshow("Face Finder", face_crops[i])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
