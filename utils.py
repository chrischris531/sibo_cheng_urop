import os
import cv2 as cv
import numpy as np

def load_videos(dataset_path):
  filepaths = os.listdir(dataset_path)
  videos = []
  for filepath in filepaths:
    cap = cv.VideoCapture(dataset_path + filepath)
    video = []

    while cap.isOpened():
      # Capture frame-by-frame
      ret, frame = cap.read()

      # if frame is read correctly ret is True
      if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

      frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY) 

      video.append(frame)

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()

    video = np.array(video)

    videos.append(video)

  videos = np.array(videos)
  video_labels = [os.path.splitext(filepath)[0] for filepath in filepaths]

  return videos, video_labels

def load_video_save_frames(dataset_path, target_path):
  # get training videos
  videos, video_labels = load_videos(dataset_path)

  # for every training video
  for i in range(len(videos)):
    # get the video and its label
    video, video_label = videos[i], video_labels[i]

    if not os.path.exists(f'{target_path}/{video_label}'):
      os.makedirs(f'{target_path}/{video_label}')

    # for every frame in each video
    for j in range(len(video)):
      # get the frame
      frame = video[j]

      # save the frame as a new image with name = index inside folder with name = video 
      cv.imwrite(f'{target_path}/{video_label}/{j}.png', frame)