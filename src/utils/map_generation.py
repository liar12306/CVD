import os
import sys
import sysconfig

import scipy.io as scio
import pandas as pd
sys.path.append("../..")
from tqdm import tqdm
from MSTmap_generation.create_map import *


def map_generation():
    #读取已经处理好的视频集合
    # processed = set()
    # data_path = config.PROJECT_ROOT + config.DATA_PATH
    # if os.path.exists(data_path):
    #     data_list = os.listdir(data_path)

    #     for dir_name in data_list:
    #         video = dir_name.split("_")[0:-1]
    #         prefix = ""
    #         for item in video:
    #             prefix += (item + "_")
    #         processed.add(prefix)
    # 打开视频list，提取路径
    with open(config.video_data_list_file) as f:

        for line in tqdm(f.readlines()):
            # 读取视频的列表文件
            path = line.strip("\n")
            # p1_v1_source1_
            prefix = path.replace("/", "_")
            # 如果处理过就跳过
            # if prefix in processed:
            #     continue
            video_dir = config.video_path + path
            #create_map(video_dir, prefix)

            video_path = video_dir + "video.avi"
            
            if not os.path.exists(video_path):
                continue
            # 获取每个视频人脸关键点数据并保存
            # 获取视频帧
            frames, fps = get_frames_and_video_meta_data(video_path)
            landmarks_path = video_dir+"landmarks68/"
            if not os.path.exists(landmarks_path):
                    os.mkdir(landmarks_path)
            else:
                print(video_dir)
                continue
            for idx, frame in enumerate(frames):
                landmarks = get_faces_landmarks(frame)
                landmarks = np.swapaxes(landmarks,1,0)
                scio.savemat(landmarks_path+"{}.mat".format(idx), {"landmarks": landmarks})



if __name__ == "__main__":
    map_generation()


