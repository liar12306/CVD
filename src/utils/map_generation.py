import sys
sys.path.append("../..")
from tqdm import tqdm
from MSTmap_generation.create_map import *
def map_generation():
    processed = set()
    if os.path.exists(config.PROJECT_ROOT+config.train_data_paths):

        with open(config.PROJECT_ROOT+config.train_data_paths,"r") as f:
            for line in f.readlines():
                video = line.split("/")[-1].split(".")[0].split("_")[0:-1]
                prefix = ""
                for item in video:
                    prefix += (item+"_")
                processed.add(prefix)
    with open(config.video_data_list_file) as f:


        for line in tqdm(f.readlines()):

            # 读取视频的列表文件
            path = line.strip("\n")
            prefix = path.replace("/", "_")
            if prefix in processed:
                continue
            video_dir = config.video_path+path

            create_map(video_dir, prefix)




if __name__ == "__main__":
    map_generation()

