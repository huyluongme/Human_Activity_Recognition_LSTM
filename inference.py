import cv2
import mediapipe as mp
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from keras.models import load_model
import threading

physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_visible_devices(physical_devices[0], 'GPU')

num_of_timesteps = 7
model = load_model(f'model/model_{num_of_timesteps}.h5')

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils


def make_landmark_timestep(results):
    lm_list = []
    landmarks = results.pose_landmarks.landmark
    
    base_x = landmarks[0].x
    base_y = landmarks[0].y
    base_z = landmarks[0].z
    
    center_x = np.mean([lm.x for lm in landmarks])
    center_y = np.mean([lm.y for lm in landmarks])
    center_z = np.mean([lm.z for lm in landmarks])

    distances = [np.sqrt((lm.x - center_x)**2 + (lm.y - center_y)**2 + (lm.z - center_z)**2) for lm in landmarks[1:]]

    scale_factors = [1.0 / dist for dist in distances]

    lm_list.append(0.0)
    lm_list.append(0.0)
    lm_list.append(0.0)
    lm_list.append(landmarks[0].visibility)

    for lm, scale_factor in zip(landmarks[1:], scale_factors):
        lm_list.append((lm.x - base_x) * scale_factor)
        lm_list.append((lm.y - base_y) * scale_factor)
        lm_list.append((lm.z - base_z) * scale_factor)
        lm_list.append(lm.visibility)
    return lm_list

def draw_landmark_on_image(results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    h, w, c = img.shape
    bbox = []
    if results.pose_landmarks:
        for id, lm in enumerate(results.pose_landmarks.landmark):
            cx, cy = int(lm.x * w), int(lm.y * h)
            bbox.append([cx, cy])
        x_min, y_min = np.min(bbox, axis=0)
        x_max, y_max = np.max(bbox, axis=0)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
    return img

def draw_class_on_image(label, img):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (20, 50)
    fontScale = 1
    fontColor = (0, 255, 0)
    thickness = 2
    lineType = 2
    cv2.putText(img, label,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                thickness,
                lineType)
    return img

label = "Unknown"

def detect(model, lm_list):
    global label
    lm_list = np.array(lm_list)
    lm_list = np.expand_dims(lm_list, axis=0)
    results = model.predict(lm_list)
    predicted_label_index = np.argmax(results, axis=1)[0]
    classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']
    confidence = np.max(results, axis=1)[0]
    if confidence > 0.95:
        label = classes[predicted_label_index]
    else:
        label = "neutral"


cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

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
        frame = draw_landmark_on_image(results, frame)
    # frame = cv2.flip(frame, 1)
    frame = draw_class_on_image(label, frame)
    cv2.imshow("image", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()
