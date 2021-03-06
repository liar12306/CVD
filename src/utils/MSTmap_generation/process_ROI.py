from numpy.linalg import linalg

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
    #frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
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
    landx = landmarks[:,0]
    landy = landmarks[:,1]
    # cv2.rectangle(frame, (face.x, face.y), (face.x + face.width, face.y + face.height), (255, 0, 0), 1)
    #
    # for landmark in landmarks:
    #     cv2.circle(frame, (landmark[0], landmark[1]), 1, (255, 0, 0), -1)
    #
    face_x = face.x
    face_y = face.y
    face_h = face.height
    face_w = face.width
    # landmarks[:, 0] = landmarks[:, 0] - face_x
    # landmarks[:, 1] = landmarks[:, 1] - face_y
    # cv2.circle(frame, (face_x, face_y), 10, (0, 255, 0), -1)
    # faces.append(frame[face.y:face.y+face.height, face.x:face.x+face.width,:])
    # #cv2.rectangle(frame, (face.x, face.y), (face_h, face_w), (255, 0, 0), 2)
    # plt.figure()
    # plt.imshow(frame[face_y:face_w, face_x:face_h,:])
    # #plt.imshow(frame)
    # plt.show()
    landmarks[:,0:1] = landmarks[:,0:1]-face.x
    landmarks[:,1:] = landmarks[:,1:]-face.y
    return frame[face_y:face_y+face_h, face_x:face_x+face_w, :], landmarks


def process_ROI(face, landmarks):
    h, w, c = face.shape

    ROI = {
        "left1": [0, 1, 2, 31, 41, 0],
        "left2": [2, 3, 4, 5, 48, 31, 2],
        "right1": [16, 15, 14, 35, 46, 16],
        "right2": [14, 13, 12, 11, 54, 35, 14],
        "mouth": [5, 6, 7, 8, 9, 10, 11, 54, 56, 57, 58, 48, 5],

    }
    forehead_lmks_idx = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    forehead = landmarks[forehead_lmks_idx]
    l = np.mean(landmarks[36:42], axis=0)
    r = np.mean(landmarks[42:48], axis=0)
    e_dis = linalg.norm(l-r)
    tmp = (np.mean(landmarks[17:22], axis=0)+np.mean(landmarks[22:27], axis=0))/2 - (l+r)/2
    tmpp = e_dis/linalg.norm(tmp)*0.6*tmp
    ROI["forehead"] = list(forehead)
    ROI["forehead"].append(np.array(forehead[-1])+tmpp)
    ROI["forehead"].append(np.array(forehead[-0])+tmpp)
    ROI["forehead"].append(np.array(forehead[0]))

    rois = []
    roi_pix_nums = []
    for key in ROI:
        mask = np.zeros(face.shape, dtype="uint8")
        # ??????
        if key == "forehead":
            cv2.fillPoly(mask, np.array([ROI[key]], dtype=np.int32), (255, 255, 255))
        else:
            cv2.fillPoly(mask, np.array([landmarks[ROI[key]]], dtype=np.int32), (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)


        # ??????roi????????????
        roi_pix_nums.append((np.bincount(mask.flatten())[255]))


        roi = np.zeros((h, w, 6))
        roi[:, :, 0:3] = cv2.bitwise_and(face, face, mask=mask)

        yuvface = cv2.cvtColor(face, cv2.COLOR_RGB2YUV)
        roi[:, :, 3:6] = cv2.bitwise_and(yuvface, yuvface, mask=mask)
        rois.append(roi)

    return rois, np.array(roi_pix_nums)


if __name__ == "__main__":
    path = config.PROJECT_ROOT + "/data/img.jpg"
    print(path)
    img = cv2.imread(path)

    landmarks, face = get_faces_landmarks(img)


    rois = process_ROI(face, landmarks)
