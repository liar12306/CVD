from src import config
from MSTmap_generation.create_map import *

def map_generation():
    with open(config.video_data_list_file) as f:
        for line in f.readlines():
            # 读取视频的列表文件
            path = line.strip("\n")
            save_dir = path.replace("/" , "_")
            video_dir = config.video_path+path
            time_file_path = video_dir + "time.txt"
            print(save_dir)
            time_data = []
            with open(time_file_path, "r") as file:
                for line in file.readlines():
                    time_data.append(int(line.strip('\n')))
            if (time_data[-1] - time_data[0]) / 1000 < 30:
                continue

            create_map(video_dir, save_dir)


if __name__ == "__main__":
    map_generation()
