import sys
sys.path.append("../..")
from tqdm import tqdm
from MSTmap_generation.create_map import *
def map_generation():
    processed_video_list = set()
    with open(config.PROJECT_ROOT+config.train_data_paths,"r") as f:
        for line in f.readlines():
            video = line.split("/")[-1].split(".")[0].split("_")[:-1]
            tmp = ""
            for item in video:
                tmp = tmp+item+"_"
            processed_video_list.add(tmp)


    with open(config.video_data_list_file) as f:

        for line in tqdm(f.readlines()):
            # 读取视频的列表文件
            path = line.strip("\n")
            prefix = path.replace("/", "_")

            if prefix in processed_video_list:

                continue
            video_dir = config.video_path+path
            time_file_path = video_dir + "time.txt"
            if not os.path.exists(time_file_path):
                continue
            create_map(video_dir, prefix)


if __name__ == "__main__":
     map_generation()

