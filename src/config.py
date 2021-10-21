import os
PROJECT_ROOT = os.path.abspath(os.path.dirname(os.path.dirname(__file__))).replace('\\','/')
with open(PROJECT_ROOT+"/path.txt","r") as f:
    video_path = f.readline().strip('\n')
DATA_PATH = "/data/train/"
ROI_NUM = 5
ROI_COMBINATION_NUM = 31
MAP_CHANEL_NUM = 6
CLIP_LENGTH = 300

video_data_list_file = video_path+"test.txt"
train_data_paths = "/data/train.txt"
