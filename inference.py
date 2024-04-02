import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.utils import to_categorical
import threading

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

model = load_model('model/model.h5')

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

num_of_timesteps = 5

def make_landmark_timestep(results):
    lm_list = []
    landmarks = results.pose_landmarks.landmark
    
    # Lấy tọa độ của điểm cố định (landmark có id 0)
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z
    
    # Tính toán tọa độ trung tâm của bàn tay
    center_x = np.mean([lm.x for lm in landmarks])
    center_y = np.mean([lm.y for lm in landmarks])
    center_z = np.mean([lm.z for lm in landmarks])

    # Tính toán khoảng cách từ trung tâm đến mỗi landmark (trừ landmark có id 0) và lưu vào một list
    distances = [np.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2 + (lm.z - center_z)**2) for lm in landmarks[1:]]

    # Tính toán tỉ lệ scale cho mỗi landmark
    scale_factors = [1.0 / dist for dist in distances]

    # Scale và trừ tọa độ của các landmark sao cho chúng có cùng một phạm vi
    # Trừ tọa độ cho landmark có id 0
    lm_list.append(0.0)  # Tọa độ x của landmark có id 0 là 0
    lm_list.append(0.0)  # Tọa độ y của landmark có id 0 là 0
    lm_list.append(0.0)  # Tọa độ z của landmark có id 0 là 0
    lm_list.append(landmarks[0].visibility)

    for lm, scale_factor in zip(landmarks[1:], scale_factors):
        lm_list.append((lm.x - base_x) * scale_factor)
        lm_list.append((lm.y - base_y) * scale_factor)
        lm_list.append((lm.z - base_z) * scale_factor)
        lm_list.append(lm.visibility)
    return lm_list

# Function to draw pose landmarks and bounding box on the image
def draw_landmarks_and_bbox(image, landmarks, bbox):
    # Draw pose landmarks
    for landmark in landmarks:
        cv2.circle(image, (int(landmark[0]), int(landmark[1])), 5, (0, 255, 0), -1)

    # Draw bounding box
    cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 255), 2)

    return image

# Function to detect pose landmarks and bounding box
def detect_landmarks_and_bbox(image):
    # Initialize MediaPipe Pose
    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # Convert the image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Detect pose landmarks
    results = pose.process(image_rgb)

    # Extract landmarks
    pose_landmarks = []
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            h, w, c = image.shape
            x, y = int(landmark.x * w), int(landmark.y * h)
            pose_landmarks.append((x, y))

    # Calculate bounding box
    bbox = [int(min([landmark[0] for landmark in pose_landmarks])),
            int(min([landmark[1] for landmark in pose_landmarks])),
            int(max([landmark[0] for landmark in pose_landmarks])),
            int(max([landmark[1] for landmark in pose_landmarks]))]

    return pose_landmarks, bbox

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    # print(lm_list.shape)
    results = model.predict(lm_list)
    print(results)
    # Chọn nhãn có giá trị dự đoán cao nhất
    predicted_label_index = np.argmax(results, axis=1)[0]
    classes = ['Fall Down', 'Lying Down', 'Sit down', 'Sitting', 'Stand up', 'Standing', 'Walking']
    label = classes[predicted_label_index]
    print(f'predicted_label_index: {predicted_label_index}')
    print(f'label: {label}')
    confidence = np.max(results, axis=1)[0]  # Tỉ lệ nhận dạng
    if confidence > 0.9:  # Kiểm tra tỉ lệ nhận dạng
        classes = ['Fall Down', 'Lying Down', 'Sit down', 'Sitting', 'Stand up', 'Standing', 'Walking']
        label = classes[predicted_label_index]
    else:
        label = "nothing"  # Nếu tỉ lệ nhận dạng không đạt 90%, gán nhãn là "nothing"
    # label_mapping = {0: "nothing", 1: "right", 2: "left", 3: "down", 4: "up"}
    # label = label_mapping[predicted_label_index]
    # sock.sendto(str.encode(label), serverAddressPort)
    # if results[0][0] > 0.5:
    #     label = "up"
    # else:
    #     label = "down"
    return label

# Initialize camera capture
cap = cv2.VideoCapture('./dataset/video/test/Fall Down/video_1.avi')
lm_list = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frameRGB)
    if results.pose_landmarks:
        lm = make_landmark_timestep(results)
        lm_list.append(lm)
        if len(lm_list) == num_of_timesteps:
            detect_thread = threading.Thread(target=detect, args=(model, lm_list,))
            detect_thread.start()
            lm_list = []

    # ret, frame = cap.read()
    #     frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #     results = pose.process(frameRGB)
    #     if results.pose_landmarks:
    #         lm = make_landmark_timestep(results)
    #         lm_list.append(lm)
    #         if len(lm_list) == num_of_timestep:
    #             d
    # # Read frame from camera
    # ret, frame = cap.read()
    # if not ret:
    #     break

    # # Detect pose landmarks and bounding box
    # pose_landmarks, bbox = detect_landmarks_and_bbox(frame)

    # # Draw landmarks and bounding box on the frame
    # frame_with_landmarks_and_bbox = draw_landmarks_and_bbox(frame.copy(), pose_landmarks, bbox)

    # # Display the frame with landmarks and bounding box
    # cv2.imshow("Pose Landmarks and Bounding Box", frame_with_landmarks_and_bbox)

    # # Press 'q' to exit
    # if cv2.waitKey(1) & 0xFF == ord('q'):
    #     break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
