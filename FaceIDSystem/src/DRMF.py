import cv2
import dlib, os
import numpy as np
from sklearn.ensemble import RandomForestRegressor


MODEL_PATH=os.path.join('FaceIDSystem','src','models','DRMF.dat')
REG_IMG=os.path.join('FaceIDSystem','src','img','reg_img.jpg')
TEST_IMG=os.path.join('FaceIDSystem','src','img','Tavkevic.jpg')
OUTPUT_IMG=os.path.join('FaceIDSystem','src','img','output_asm.jpg')

class DRMFFaceLandmarker:
    def __init__(self, model_path=MODEL_PATH):
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(model_path)
        self.feature_extractor = self._create_feature_extractor()

    def _create_feature_extractor(self):
        # Пример фильтров для извлечения признаков
        filters = [
            cv2.filter2D(np.eye(3), -1, np.array([[-1,0,1]])),  # Sobel X
            cv2.filter2D(np.eye(3), -1, np.array([[-1],[0],[1]]))  # Sobel Y
        ]
        return filters

    def extract_features(self, patch: np.ndarray) -> np.ndarray:
        return np.concatenate([f(patch).flatten() for f in self.feature_extractor])

    def detect_landmarks(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.detector(gray)
        
        if not faces:
            return None
            
        landmarks = self.predictor(gray, faces[0])
        return np.array([(p.x, p.y) for p in landmarks.parts()])

    def train_custom_model(self, training_data):
        X, y = [], []
        for img_path, true_points in training_data:
            detected = self.detect_landmarks(img_path)
            if detected is not None:
                patch = self._extract_patches(cv2.imread(img_path), detected)
                X.append(self.extract_features(patch))
                y.append(true_points - detected)  # Residuals
        
        self.refinement_model = RandomForestRegressor(n_estimators=100)
        self.refinement_model.fit(X, y)

# Пример использования
drmf = DRMFFaceLandmarker()
landmarks = drmf.detect_landmarks(REG_IMG)
print(f"DRMF detected {len(landmarks)}")

