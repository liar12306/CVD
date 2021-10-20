import cv2
import numpy as np
from src import config


def get_frame_rate(cap):
    return cap.get(5)


def get_frame_W_And_H(cap):
    return int(cap.get(3)), int(cap.get(4))


def get_frames(cap, H, W):
    num_frames = int(cap.get(7))

    # Frames from the video have shape NumFrames x H x W x C
    frames = np.zeros((num_frames, H, W, 3), dtype='uint8')

    frame_counter = 0
    while cap.isOpened():
        # curr_frame_id = int(cap.get(1))  # current frame number
        ret, frame = cap.read()
        if not ret:
            break

        frames[frame_counter, :, :, :] = frame
        frame_counter += 1
        if frame_counter == num_frames:
            break
    return frames


def get_slideing_window_stride(frameRate):
    return int(frameRate / 2)


def get_frame_num(cap):
    return int(cap.get(7))


def get_frames_and_video_meta_data(video_path, meta_data_only=False):
    cap = cv2.VideoCapture(video_path)
    frameRate = get_frame_rate(cap)  # frame rate

    # Frame dimensions: WxH
    (W, H) = get_frame_W_And_H(cap)

    # Paper mentions a stride of 0.5 seconds = 15 frames
    sliding_window_stride = get_slideing_window_stride(frameRate)

    num_frames = get_frame_num(cap)

    if meta_data_only:
        return {"frame_rate": frameRate, "sliding_window_stride": sliding_window_stride, "num_frames": num_frames}

    # Frames from the video have shape NumFrames x H x W x C
    frames = get_frames(cap, H, W)

    return frames

if __name__ == "__main__":
    video_path = config.PROJECT_ROOT+config.DATA_PATH+"video.avi"
    data = get_frames_and_video_meta_data(video_path, True)
    # print("frameRate:{}\nsliding_window_stride: {}\nnum_frames: {}\n".format(data["frame_rate"],data["sliding_window_stride"],data["num_frames"]))
