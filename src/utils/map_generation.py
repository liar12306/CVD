import sys
sys.path.append("../..")
from tqdm import tqdm
from MSTmap_generation.create_map import *
def map_generation():
    with open(config.video_data_list_file) as f:

        for line in tqdm(f.readlines()):

            # 读取视频的列表文件
            path = line.strip("\n")
            prefix = path.replace("/" , "_")
            video_dir = config.video_path+path
            time_file_path = video_dir + "time.txt"
            if not os.path.exists(time_file_path):
                continue
            create_map(video_dir, prefix)


if __name__ == "__main__":
    map_generation()
