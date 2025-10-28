import cv2
import numpy as np
from skimage.feature import hog
import joblib

svm_model = joblib.load("models/gesture_svm_model.pkl")
classes = joblib.load("models/gesture_classes.pkl")

def extract_hog_features(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    features = hog(
        gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm='L2-Hys'
    )
    return features.reshape(1, -1)  

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Cannot open webcam")
    exit()

print("Press 'q' to quit")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    
    img = cv2.resize(frame, (64, 64))

    
    features = extract_hog_features(img)
    pred = svm_model.predict(features)[0]
    gesture_name = classes[pred]
  
    cv2.putText(frame, f"Gesture: {gesture_name}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Hand Gesture Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
