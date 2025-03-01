import numpy as np
import cv2 as cv
import os
import json
from ultralytics import YOLO

model = YOLO("yolo11n-pose.pt")

image_folder = "IMG1/"  
output_file = "dataset3.json"

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]

ANGLE_JOINTS = [
    (5, 6, 8),   
    (6, 5, 7),   
    (11, 12, 14), 
    (12, 11, 13)  
]

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)

    ba = a - b
    bc = c - b

    cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
    angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))  # تبدیل به درجه
    return np.degrees(angle)

dataset = []

for image_name in os.listdir(image_folder):
    image_path = os.path.join(image_folder, image_name)
    
    img = cv.imread(image_path)
    if img is None:
        print(f"Error loading {image_name}")
        continue

    results = model(img)
    key_points = results[0].keypoints.data.cpu().numpy()

    image_data = {
        "image_name": image_name,
        "keypoints": [],
        "connections": [],
        "angles": {}
    }

    for person in key_points:
        person_keypoints = []

        for key_point in person:
            key_point = key_point[:2].astype(np.uint16)
            person_keypoints.append(key_point.tolist())

        image_data["keypoints"].append(person_keypoints)

        for connection in SKELETON:
            kp1_idx, kp2_idx = connection
            kp1 = person_keypoints[kp1_idx]
            kp2 = person_keypoints[kp2_idx]
            image_data["connections"].append([kp1, kp2])

        for joint in ANGLE_JOINTS:
            p1, p2, p3 = joint
            angle = calculate_angle(person_keypoints[p1], person_keypoints[p2], person_keypoints[p3])
            image_data["angles"][f"{p1}-{p2}-{p3}"] = angle

    dataset.append(image_data)

with open(output_file, "w") as f:
    json.dump(dataset, f, indent=4)

print(f"Dataset saved to {output_file}")
