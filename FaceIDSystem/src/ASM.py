import cv2,os
import dlib
import numpy as np


MODEL_PATH=os.path.join('FaceIDSystem','src','shape_predictor_68_face_landmarks.dat')
REG_IMG=os.path.join('FaceIDSystem','src','img','reg_img.jpg')
TEST_IMG=os.path.join('FaceIDSystem','src','img','test_img.jpg')
OUTPUT_IMG=os.path.join('FaceIDSystem','src','img','output_asm.jpg')


class ASMFaceLandmarker:
    def __init__(self, model_path=MODEL_PATH):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)

    def detect_landmarks(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Image not loaded")
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if not faces:
            return None
            
        landmarks = self.predictor(gray, faces[0])
        return np.array([(p.x, p.y) for p in landmarks.parts()])

    def visualize(self, image_path: str, output_path: str):
        img = cv2.imread(image_path)
        points = self.detect_landmarks(image_path)
        
        if points is not None:
            for (x, y) in points:
                cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        
        cv2.imwrite(output_path, img)

# Пример использования
asm = ASMFaceLandmarker()
landmarks = asm.detect_landmarks(REG_IMG)
print(f"Detected {len(landmarks)} points" if landmarks is not None else "No face found")
asm.visualize(REG_IMG, OUTPUT_IMG)