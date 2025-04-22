import cv2, os
import numpy as np
from cv2 import face

MODEL_PATH=os.path.join('FaceIDSystem','src','models','lbfmodel.yaml')
REG_IMG=os.path.join('FaceIDSystem','src','img','reg_img.jpg')
TEST_IMG=os.path.join('FaceIDSystem','src','img','Tavkevic.jpg')
OUTPUT_IMG=os.path.join('FaceIDSystem','src','img','output_asm.jpg')


class CLMFaceLandmarker: # Note: The class name suggests CLM, but the model is LBF. Keep this in mind.
    def __init__(self, model_path=MODEL_PATH):
        # Create a FacemarkLBF object
        self.facemark = face.createFacemarkLBF()
        # Load the trained model
        print(f"loading data from : {model_path}")
        self.facemark.loadModel(model_path)
        # Initialize face detector
        self.detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def detect_landmarks(self, image_path: str) -> np.ndarray:
        """
        Detects face and then fits the LBF model to find landmarks.
        The fit method itself performs the optimization/refinement for LBF.
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"Error loading image: {image_path}")
            return None

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect faces
        faces = self.detector.detectMultiScale(gray)
        if len(faces) == 0:
            print(f"No faces detected in {image_path}")
            return None

        # Fit the LBF model to the detected faces
        # The fit method returns a success flag and a list of landmarks
        # (one list per detected face)
        success, landmarks = self.facemark.fit(gray, np.array(faces))

        if success:
            # landmarks is a list, even if only one face was detected.
            # It contains numpy arrays of shape (num_landmarks, 2).
            # We return the landmarks for the first detected face.
            return landmarks[0]
        else:
            print(f"Facemark fitting failed for {image_path}")
            return None

    # The optimize_landmarks method using 'refine' is not applicable for FacemarkLBF
    # def optimize_landmarks(self, image_path: str, init_points: np.ndarray) -> np.ndarray:
    #     # This method is incorrect for FacemarkLBF as there is no 'refine' method
    #     pass


# Пример использования (Example Usage)
clm = CLMFaceLandmarker()
landmarks = clm.detect_landmarks(TEST_IMG)

if landmarks is not None:
    # The landmarks obtained from detect_landmarks are the final, fitted points
    print(f"Detected and fitted LBF points: {landmarks.shape}")

    # You can now use 'landmarks' for visualization or further processing
    # For example, drawing on the image:
    img = cv2.imread(TEST_IMG)
    if img is not None:
        for (x, y) in np.squeeze(landmarks): # np.squeeze removes the extra dimension (1, 68, 2) -> (68, 2)
            cv2.circle(img, (int(x), int(y)), 2, (0, 255, 0), -1) # Draw green circles

        cv2.imwrite(OUTPUT_IMG, img)
        print(f"Landmarks drawn on image and saved to {OUTPUT_IMG}")

else:
    print("Could not detect or fit landmarks.")