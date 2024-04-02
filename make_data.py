import os
import shutil
import cv2
import mediapipe as mp
import pandas as pd
import numpy as np

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

classes = ['Fall Down', 'Lying Down', 'Sit down', 'Sitting', 'Stand up', 'Standing', 'Walking']

if os.path.exists('./dataset/csv'):
    shutil.rmtree('./dataset/csv')

for cl in classes:
    os.makedirs(f'./dataset/csv/train/{cl}')
    os.makedirs(f'./dataset/csv/test/{cl}')

# lm_list = []
# def make_landmark_timestep(results):
#     # print(results.pose_landmarks.landmark)
#     c_lm = []
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         c_lm.append(lm.x)
#         c_lm.append(lm.y)
#         c_lm.append(lm.z)
#         c_lm.append(lm.visibility)
#     return c_lm
    
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

for cl in classes:
    cnt = 0
    lm_list = []
    for file in os.listdir(f"./dataset/video/train/{cl}"):
        print(f'processing: ./dataset/video/train/{cl}/{file}')
        cap = cv2.VideoCapture(f'./dataset/video/train/{cl}/{file}')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)
            if results.pose_landmarks:
                # Ghi nhận thông số khung xương
                lm = make_landmark_timestep(results)
                lm_list.append(lm)
        df  = pd.DataFrame(lm_list)
        df.to_csv(f'./dataset/csv/train/{cl}/{cnt}.txt')
        cnt = cnt + 1
        cap.release()
        cv2.destroyAllWindows()
        lm_list = []

    cnt = 0
    lm_list = []

    for file in os.listdir(f"./dataset/video/test/{cl}"):
        print(f'processing: ./dataset/video/test/{cl}/{file}')
        cap = cv2.VideoCapture(f'./dataset/video/test/{cl}/{file}')
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frameRGB)
            if results.pose_landmarks:
                # Ghi nhận thông số khung xương
                lm = make_landmark_timestep(results)
                lm_list.append(lm)
        df  = pd.DataFrame(lm_list)
        df.to_csv(f'./dataset/csv/test/{cl}/{cnt}.txt')
        cnt = cnt + 1
        cap.release()
        cv2.destroyAllWindows()
        lm_list = []


# cnt = 0
# for file in os.listdir(f'./dataset/video/train/Fall Down'):
#     cap = cv2.VideoCapture(f'./dataset/video/train/Fall Down/{file}')
#     print(f'./dataset/video/train/Fall Down/{file}')
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frameRGB)
#         if results.pose_landmarks:
#             # Ghi nhận thông số khung xương
#             lm = make_landmark_timestep(results)
#             lm_list.append(lm)
#         # cv2.imshow(f'./dataset/video/train/Fall Down/{file}', frame)
#     df  = pd.DataFrame(lm_list)
#     df.to_csv(f'./dataset/csv/{cnt}.txt')
#     cnt = cnt + 1
#     cap.release()
#     cv2.destroyAllWindows()
        


# # Đọc ảnh từ webcam
# cap = cv2.VideoCapture(0)

# # Khởi tạo thư viện mediapipe
# mpPose = mp.solutions.pose
# pose = mpPose.Pose()
# mpDraw = mp.solutions.drawing_utils

# lm_list = []
# label = "BODYSWING"
# no_of_frames = 600

# def make_landmark_timestep(results):
#     print(results.pose_landmarks.landmark)
#     c_lm = []
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         c_lm.append(lm.x)
#         c_lm.append(lm.y)
#         c_lm.append(lm.z)
#         c_lm.append(lm.visibility)
#     return c_lm

# def draw_landmark_on_image(mpDraw, results, img):
#     # Vẽ các đường nối
#     mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

#     # Vẽ các điểm nút
#     for id, lm in enumerate(results.pose_landmarks.landmark):
#         h, w, c = img.shape
#         print(id, lm)
#         cx, cy = int(lm.x * w), int(lm.y * h)
#         cv2.circle(img, (cx, cy), 10, (0, 0, 255), cv2.FILLED)
#     return img


# while len(lm_list) <= no_of_frames:
#     ret, frame = cap.read()
#     if ret:
#         # Nhận diện pose
#         frameRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = pose.process(frameRGB)

#         if results.pose_landmarks:
#             # Ghi nhận thông số khung xương
#             lm = make_landmark_timestep(results)
#             lm_list.append(lm)
#             # Vẽ khung xương lên ảnh
#             frame = draw_landmark_on_image(mpDraw, results, frame)

#         cv2.imshow("image", frame)
#         if cv2.waitKey(1) == ord('q'):
#             break

# # Write vào file csv
# df  = pd.DataFrame(lm_list)
# df.to_csv(label + ".txt")
# cap.release()
# cv2.destroyAllWindows()