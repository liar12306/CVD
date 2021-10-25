from src.utils.seetaface import api
from src import config
import cv2
import matplotlib.pyplot as plt
import numpy as np


def detect_faces(seetaFace, frame):
    detect_result = seetaFace.Detect(frame)
    return detect_result


def get_landmarks(seetaFace, frame, face):
    return seetaFace.mark68(frame, face)


def get_faces_landmarks(frame):
    frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    seetaFace = api.SeetaFace(api.FACE_DETECT | api.LANDMARKER68)
    faces = []
    detect_result = detect_faces(seetaFace, frame)

    # for i in range(detect_result.size):
    face = detect_result.data[0].pos
    landmark_data = get_landmarks(seetaFace, frame, face)
    landmarks = np.zeros((68, 2), dtype="int32")
    for idx in range(68):
        landmarks[idx][0] = landmark_data[idx].x
        landmarks[idx][1] = landmark_data[idx].y
    # cv2.rectangle(frame, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0), 2)

    # for landmark in landmarks:
    #     cv2.circle(frame, (landmark[0], landmark[1]), 5, (255, 0, 0), -1)

    face_x = min(face.x, landmarks[:, 0].min())
    face_y = min(face.y, landmarks[:, 1].min())
    face_h = max(face.height, landmarks[:, 0].max())
    face_w = max(face.width, landmarks[:, 1].max())
    landmarks[:, 0] = landmarks[:, 0] - face_x
    landmarks[:, 1] = landmarks[:, 1] - face_y
    # cv2.circle(frame, (face_x, face_y), 10, (0, 255, 0), -1)
    # print(landmarks)
    # print(face.x,face.height)
    # print(face.y,face.width)
    # faces.append(frame[face.y:face.y+face.height, face.x:face.x+face.width,:])
    # cv2.rectangle(frame, (face.x, face.y), (face_h, face_w), (255, 0, 0), 2)
    # plt.figure()
    # plt.imshow(frame[face_y:face_w, face_x:face_h,:])

    # plt.imshow(frame)
    # plt.show()
    # landmarks[:,0] = landmarks[:,0]-face.x
    # landmarks[:,1] = landmarks[:,1]-face.y
    return landmarks, frame[face_y:face_w, face_x:face_h, :]


def process_ROI(face, landmarks):
    h, w, c = face.shape
    roi_signal = np.zeros((5, 6))

    ROI = {
        "left1": [0, 1, 2, 31, 41, 0],
        "left2": [2, 3, 4, 5, 48, 31, 2],
        "right1": [16, 15, 14, 35, 46, 16],
        "right2": [14, 13, 12, 11, 54, 35, 14],
        "mouth": [5, 6, 7, 8, 9, 10, 11, 54, 56, 57, 58, 48, 5],
        # "forehead":[17,18,19,20,21,22,23,24,25,26]
    }

    rois = []
    roi_pix_nums = []
    for key in ROI:
        mask = np.zeros(face.shape, dtype="uint8")

        # 掩膜
        cv2.fillPoly(mask, np.array([landmarks[ROI[key]]], dtype=np.int32), (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        # 统计roi像素个数

        roi_pix_nums.append(np.bincount(mask.flatten())[255])



        roi = np.zeros((h, w, 6))
        roi[:, :, 0:3] = cv2.bitwise_and(face, face, mask=mask)

        face = cv2.cvtColor(face, cv2.COLOR_RGB2YUV)
        roi[:, :, 3:6] = cv2.bitwise_and(face, face, mask=mask)
        rois.append(roi)
    return rois, np.array(roi_pix_nums)


if __name__ == "__main__":
    path = config.PROJECT_ROOT + config.DATA_PATH + "img.png"
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    landmarks, face = get_faces_landmarks(img)
    rois = process_ROI(face, landmarks)
