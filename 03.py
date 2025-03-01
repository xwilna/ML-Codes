import json
import numpy as np
import cv2 as cv
from ultralytics import YOLO
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

dataset_sitting = "dataset0.json"
dataset_standing = "dataset1.json"

def load_data(json_file, label):
    with open(json_file, "r") as f:
        data = json.load(f)

    X, y = [], []
    for item in data:
        angles = item["angles"]
        feature_vector = [
            angles["5-6-8"],
            angles["6-5-7"],
            angles["11-12-14"],
            angles["12-11-13"]
        ]
        X.append(feature_vector)
        y.append(label)
    
    return X, y

X_sitting, y_sitting = load_data(dataset_sitting, label=0)
X_standing, y_standing = load_data(dataset_standing, label=1)

X = np.array(X_sitting + X_standing)
y = np.array(y_sitting + y_standing)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

yolo_model = YOLO("yolo11n-pose.pt")

ANGLE_JOINTS = [
    (5, 6, 8),
    (6, 5, 7),
    (11, 12, 14),
    (12, 11, 13)
]

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    ba = a - b
    bc = c - b
    
    norm_ba = np.linalg.norm(ba)
    norm_bc = np.linalg.norm(bc)

    if norm_ba == 0 or norm_bc == 0: 
        return 0.0

    cosine_angle = np.dot(ba, bc) / (norm_ba * norm_bc)
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))
    return np.degrees(angle)

def predict_pose(keypoints):
    angles = []
    
    for p1, p2, p3 in ANGLE_JOINTS:
        if np.any(np.isnan([keypoints[p1][:2], keypoints[p2][:2], keypoints[p3][:2]])):
            return "Unknown" 
        
        angle = calculate_angle(keypoints[p1][:2], keypoints[p2][:2], keypoints[p3][:2])
        angles.append(angle)

    feature_vector = np.array(angles).reshape(1, -1)

    
    if np.any(np.isnan(feature_vector)):
        return "Unknown"

    prediction = knn.predict(feature_vector)
    return "Standing" if prediction[0] == 1 else "Sitting"

cap = cv.VideoCapture(0) 

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = yolo_model(frame)
    keypoints_data = results[0].keypoints.data.cpu().numpy() if results[0].keypoints is not None else []

    for keypoints in keypoints_data:
        keypoints = keypoints[:, :2] 

        pose = predict_pose(keypoints)

        x_min, y_min = np.min(keypoints, axis=0).astype(int)
        x_max, y_max = np.max(keypoints, axis=0).astype(int)

        cv.rectangle(frame, (x_min, y_min), (x_max, y_max), (255,255,255), 2)
        text_size = cv.getTextSize(pose, cv.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        text_x = max(x_min, x_min + (x_max - x_min - text_size[0]) // 2)
        text_y = max(y_min - 10, 20)  
        cv.putText(frame, pose, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2) 

    cv.imshow("Live Pose Detection", frame)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv.destroyAllWindows()
