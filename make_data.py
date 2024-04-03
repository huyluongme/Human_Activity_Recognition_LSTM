import os
import shutil
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

classes = ['boxing', 'handclapping', 'handwaving', 'jogging', 'running', 'walking']

if os.path.exists('./dataset'):
    shutil.rmtree('./dataset')

for cl in classes:
    os.makedirs(f'./dataset/{cl}')

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

def draw_landmark_on_image(mpDraw, results, img):
    mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)
    return img

for cl in classes:
    cnt = 0
    for file in os.listdir(f'./video/{cl}'):
        lm_list = []
        print(f'cnt: {cnt}, lm_list: {lm_list}')
        print(f'processing: ./video/{cl}/{file}')
        cap = cv2.VideoCapture(f'./video/{cl}/{file}')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)
            if results.pose_landmarks:
                lm = make_landmark_timestep(results)
                lm_list.append(lm)
            #     frame = draw_landmark_on_image(mpDraw, results, frame)
            # cv2.imshow(f'processing: ./video/{cl}/{file}', frame)
            # if cv2.waitKey(1) == ord('q'):
            #     break
            
        df  = pd.DataFrame(lm_list)
        df.to_csv(f'./dataset/{cl}/{cl}_{cnt}.txt')
        cnt = cnt + 1
        cap.release()
        cv2.destroyAllWindows()