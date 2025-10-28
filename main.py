import os
import cv2
import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score
import joblib

def load_images_from_folder(folder, size=(64, 64)):
    images = []
    labels = []
    classes = sorted(os.listdir(folder))
    print("Classes found:", classes)  

    for label, class_name in enumerate(classes):
        class_path = os.path.join(folder, class_name)
        if not os.path.isdir(class_path):
            continue
        files = os.listdir(class_path)
        print(f"Class '{class_name}' has {len(files)} files")  
        for file in files:
            img_path = os.path.join(class_path, file)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, size)
                images.append(img)
                labels.append(label)
    print(f"Total images loaded from {folder}: {len(images)}")  
    return np.array(images), np.array(labels), classes


def extract_hog_features(images):
    features = []
    for i, img in enumerate(images):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hog_features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2),
            block_norm='L2-Hys'
        )
        features.append(hog_features)
        if (i + 1) % 500 == 0:
            print(f"HOG: Processed {i + 1}/{len(images)} images")
    return np.array(features)

def train_svm(train_features, train_labels):
    svm_model = SVC(kernel='linear', probability=True)
    svm_model.fit(train_features, train_labels)
    return svm_model

def evaluate_model(model, test_features, test_labels, classes):
    preds = model.predict(test_features)
    print("\nModel Evaluation Results:")
    print(f"Accuracy: {accuracy_score(test_labels, preds):.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, preds, target_names=classes))

def predict_image(model, classes, img_path):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Image not found: {img_path}")
        return
    img = cv2.resize(img, (64, 64))
    features = extract_hog_features([img])
    pred = model.predict(features)
    print(f"Predicted class for '{img_path}': {classes[pred[0]]}")

if __name__ == "__main__":

    print("Loading training data...")
    train_images, train_labels, classes = load_images_from_folder('train')
    print(f"Loaded {len(train_images)} training images.\n")

    print("Loading test data...")
    test_images, test_labels, _ = load_images_from_folder('test')
    print(f"Loaded {len(test_images)} test images.\n")

    print("Extracting HOG features from training data...")
    train_features = extract_hog_features(train_images)
    print("Extracting HOG features from test data...")
    test_features = extract_hog_features(test_images)

    print("Training SVM model...")
    svm_model = train_svm(train_features, train_labels)

    os.makedirs("models", exist_ok=True)
    joblib.dump(svm_model, "models/gesture_svm_model.pkl")
    joblib.dump(classes, "models/gesture_classes.pkl")
    print(" Model and classes saved successfully.")

    print("Evaluating model on test data...")
    evaluate_model(svm_model, test_features, test_labels, classes)

    sample_image = "test/05/sample1.png"  
    print("Predicting sample image...")
    predict_image(svm_model, classes, sample_image)

