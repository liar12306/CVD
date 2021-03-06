import os.path
import scipy.io as scio
import matplotlib.pyplot as plt
import numpy as np
from src.utils.MSTmap_generation.process_ROI import *
from src.utils.MSTmap_generation.process_video import *
from src import config
import pandas as pd
import math
import time

all_idx = []


def get_roi_signal(roi, mask_num):
    signal = np.zeros(6)
    for i in range(6):
        signal[i] = (roi[:, :, i].sum())/mask_num
    return signal


def dfs(idx, n, tmp_list):
    if idx == n:
        all_idx.append(tmp_list.copy())
        return
    tmp_list.append(0)

    dfs(idx + 1, n, tmp_list)
    tmp_list.pop()
    tmp_list.append(1)

    dfs(idx + 1, n, tmp_list)
    tmp_list.pop()


def get_signal_map(rois, roi_num, roi_pix_nums):
    # 1,1,6
    roi_signal = np.zeros((roi_num, 6))

    for idx in range(roi_num):
        roi_signal[idx, :] = get_roi_signal(rois[idx], roi_pix_nums[idx])
    dfs(0, roi_num, [])

    signal_map = np.zeros((config.ROI_COMBINATION_NUM, 6))
    for i in range(config.ROI_COMBINATION_NUM):
        idxs = all_idx[i + 1]
        signal = np.zeros(6)
        pix_sum = (idxs * roi_pix_nums).sum()
        for idx, val in enumerate(idxs):
            if val == 1:
                tmp = roi_pix_nums[idx] / pix_sum
                signal = signal + roi_signal[idx] * tmp
        signal_map[i, :] = signal
    return signal_map


def create_map(video_dir, prefix):
    video_path = video_dir + "video.avi"
    if not os.path.exists(video_path):
        return

    # 获取视频帧
    frames, fps = get_frames_and_video_meta_data(video_path)
    st_map = np.zeros((len(frames), config.ROI_COMBINATION_NUM, config.MAP_CHANEL_NUM))
    faces = []
    flandmarks = []

    for idx, frame in enumerate(frames):

        landmarks = get_faces_landmarks(frame)
        try:
            rois, roi_pix_nums = process_ROI(frame, landmarks)
        except:
            if not os.path.exists(config.PROJECT_ROOT+config.DATA_PATH):
                os.mkdir(config.PROJECT_ROOT+config.DATA_PATH)
            with open(config.PROJECT_ROOT + config.DATA_PATH + "fail.txt", "a+") as f:
                f.write(prefix + "\n")
            return
        st_map[idx, :, :] = get_signal_map(rois, config.ROI_NUM, roi_pix_nums)

    st_map = np.swapaxes(st_map, 0, 1)

    for idx in range(config.ROI_COMBINATION_NUM):
        for c in range(config.MAP_CHANEL_NUM):
            tmp_channel_data = st_map[idx, :, c]
            minn = tmp_channel_data.min()
            maxx = tmp_channel_data.max()
            st_map[idx, :, c] = (tmp_channel_data - minn) / (maxx - minn) * 255

    save_maps(st_map, video_dir, prefix, fps)


def save_maps(st_map, video_dir, prefix, fps):
    gt_hr_file_path = video_dir + "gt_HR.csv"
    BVP_file_path = video_dir + "wave.csv"
    clip_length = config.CLIP_LENGTH

    # 读取心率数据
    gt_data = pd.read_csv(gt_hr_file_path)["HR"].to_numpy()

    # 读取BVP数据
    bvp_data = pd.read_csv(BVP_file_path)["Wave"].to_numpy()

    frame_num = st_map.shape[1]
    # 每0.5s提取一次300帧长度
    clip_num = int((frame_num - clip_length) / fps * 2)
    save_dir = config.PROJECT_ROOT + config.DATA_PATH
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for idx in range(clip_num):
        start_idx = int(0.5 * fps * idx)
        if start_idx + clip_length - 1 > frame_num:
            break
        if int(start_idx / fps) >= len(gt_data):
            break
        end_idx = start_idx + clip_length - 1

        # start_idx 到 end_idx 这个区间的平均心率
        gt_tmp = np.mean(gt_data[max(1, int(start_idx / fps)): min(len(gt_data), int((end_idx + 1) / fps))])

        # map片段
        map_clip = st_map[:, start_idx:end_idx, :].astype(np.uint8)

        # 每分钟心跳次数

        bpm = (gt_tmp/60) * (clip_length / fps)
        #bpm = gt_tmp
        #
        bvp_begin = int(start_idx / frame_num * len(bvp_data))
        bvp_len = math.ceil(clip_length / frame_num * len(bvp_data))

        if bvp_begin + bvp_len > len(bvp_data):
            break
        bvp = bvp_data[bvp_begin: bvp_begin + bvp_len]
        x = np.array(range(bvp_len))
        xx = np.array(range(clip_length))
        xx = xx * bvp_len / clip_length

        bvp = np.interp(xx, x, bvp)

        train_data = {
            "rgb_map": map_clip[:, :, 0:3],
            "yuv_map": map_clip[:, :, 3:6],
            "fps": fps,
            "bpm": bpm,
            "bvp": bvp,
        }

        save_path = save_dir + "/{}{}".format(prefix, idx)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
            scio.savemat(save_path + "/fps.mat", {'fps': train_data['fps']})
            scio.savemat(save_path + "/bvp.mat", {'bvp': train_data['bvp']})
            scio.savemat(save_path + "/bpm.mat", {'bpm': train_data['bpm']})
            cv2.imwrite(save_path + "/rgb_map.png", train_data['rgb_map'])
            cv2.imwrite(save_path + "/yuv_map.png", train_data['yuv_map'])

def process_skin(face, landmarks):
    skin = []
    h, w, c = face.shape
    # le_x1 = np.min(landmarks[17:22,:1])
    # le_y1 = np.min(landmarks[17:22,1:])
    # le_x2 = np.max(landmarks[17:22,:1])
    # le_y2 = np.max(landmarks[36:42, 1:])+5
    #
    # re_x1 = np.min(landmarks[22:27, :1])
    # re_y1 = np.min(landmarks[22:27, 1:])
    # re_x2 = np.max(landmarks[22:27, :1])
    # re_y2 = np.max(landmarks[42:48, 1:]) + 5
    #
    # me_x1 = np.min(landmarks[48:69, :1])
    # me_y1 = np.min(landmarks[48:69, 1:])
    # me_x2 = np.max(landmarks[48:69, :1])
    # me_y2 = np.max(landmarks[48:69, 1:]) + 5
    # face[re_y1:re_y2,re_x1:re_x2] = 0
    # face[le_y1:le_y2, le_x1:le_x2] = 0
    # face[me_y1:me_y2, me_x1:me_x2] = 0

    ROI = {
        "skin": [0, 1, 2, 3, 4, 5,6,7,8,9,10,11,12,13,14,15,16,26,25,24,23,22,21,20,19,18,17],
        "l":[36,37,38,39,40,41],
        "r":[42,43,44,45,46,47],
        "m":[48,49,50,51,52,53,54,55,56,57,58,59,60]
    }
    forehead_lmks_idx = [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
    forehead = landmarks[forehead_lmks_idx]
    l = np.mean(landmarks[36:42], axis=0)
    r = np.mean(landmarks[42:48], axis=0)
    e_dis = linalg.norm(l - r)
    tmp = (np.mean(landmarks[17:22], axis=0) + np.mean(landmarks[22:27], axis=0)) / 2 - (l + r) / 2
    tmpp = e_dis / linalg.norm(tmp) * 0.6 * tmp
    ROI["forehead"] = list(forehead)
    ROI["forehead"].append(np.array(forehead[-1]) + tmpp)
    ROI["forehead"].append(np.array(forehead[-0]) + tmpp)
    ROI["forehead"].append(np.array(forehead[0]))
    rois = []
    for key in ROI:
        mask = np.zeros(face.shape, dtype="uint8")
        # 掩膜
        if key == "forehead":
            cv2.fillPoly(mask, np.array([ROI[key]], dtype=np.int32), (255, 255, 255))
        else:
            cv2.fillPoly(mask, np.array([landmarks[ROI[key]]], dtype=np.int32), (255, 255, 255))
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        roi = np.zeros((h, w, 3),dtype="uint8")
        roi[:, :, :] = cv2.bitwise_and(face, face, mask=mask)
        rois.append(roi)
    target = rois[0]-rois[1]-rois[2]-rois[3]+rois[4]


    plt.imshow(target)
    plt.show()



if __name__ == "__main__":
    start_time = time.time()
    path = "F:/python/rppg/data/VIPL/p1/v1/source1/video.avi"
    frames, fps = get_frames_and_video_meta_data(path)
    frame = cv2.cvtColor(frames[2], cv2.COLOR_BGR2RGB)
    face, landmarks = get_faces_landmarks(frame)
    process_skin(face,landmarks)
    end_time = time.time()
    cost = int(end_time - start_time)
    print("\n{} m : {} s".format(int(cost / 60), cost % 60))
