import os
import subprocess
import numpy as np
import cv2
import json
import dlib
import skvideo.io
from tqdm import tqdm
from collections import deque
from skimage import transform as tf

from concurrent.futures import ProcessPoolExecutor, as_completed

np.float = np.float64
np.int = np.int_
FACE_PREDICTOR_PATH = 'shape_predictor_68_face_landmarks.dat'
MEAN_FACE_PATH = '20words_mean_face.npy'
STD_SIZE = (256, 256)
STABLE_PNTS_IDS = [33, 36, 39, 42, 45]


def linear_interpolate(landmarks, start_idx, stop_idx):
    start_landmarks = landmarks[start_idx]
    stop_landmarks = landmarks[stop_idx]
    delta = stop_landmarks - start_landmarks
    for idx in range(1, stop_idx - start_idx):
        landmarks[start_idx + idx] = start_landmarks + idx / float(stop_idx - start_idx) * delta
    return landmarks


def landmarks_interpolate(landmarks):
    """Interpolate landmarks
    param list landmarks: landmarks detected in raw videos
    """

    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    if not valid_frames_idx:
        return None
    for idx in range(1, len(valid_frames_idx)):
        if valid_frames_idx[idx] - valid_frames_idx[idx - 1] == 1:
            continue
        else:
            landmarks = linear_interpolate(landmarks, valid_frames_idx[idx - 1], valid_frames_idx[idx])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    # -- Corner case: keep frames at the beginning or at the end failed to be detected.
    if valid_frames_idx:
        landmarks[:valid_frames_idx[0]] = [landmarks[valid_frames_idx[0]]] * valid_frames_idx[0]
        landmarks[valid_frames_idx[-1]:] = [landmarks[valid_frames_idx[-1]]] * (len(landmarks) - valid_frames_idx[-1])
    valid_frames_idx = [idx for idx, _ in enumerate(landmarks) if _ is not None]
    assert len(valid_frames_idx) == len(landmarks), "not every frame has landmark"
    return landmarks


def apply_transform(transform, img, std_size):
    warped = tf.warp(img, inverse_map=transform.inverse, output_shape=std_size)
    warped = warped * 255
    warped = warped.astype('uint8')
    return warped


def cut_patch(img, landmarks, height, width, threshold=5):
    center_x, center_y = np.mean(landmarks, axis=0)

    if center_y - height < 0:
        center_y = height
    if center_y - height < 0 - threshold:
        raise Exception('too much bias in height')
    if center_x - width < 0:
        center_x = width
    if center_x - width < 0 - threshold:
        raise Exception('too much bias in width')

    if center_y + height > img.shape[0]:
        center_y = img.shape[0] - height
    if center_y + height > img.shape[0] + threshold:
        raise Exception('too much bias in height')
    if center_x + width > img.shape[1]:
        center_x = img.shape[1] - width
    if center_x + width > img.shape[1] + threshold:
        raise Exception('too much bias in width')

    cutted_img = np.copy(img[int(round(center_y) - round(height)): int(round(center_y) + round(height)),
                         int(round(center_x) - round(width)): int(round(center_x) + round(width))])
    return cutted_img


def warp_img(src, dst, img, std_size):
    tform = tf.estimate_transform('similarity', src, dst)  # find the transformation matrix
    warped = tf.warp(img, inverse_map=tform.inverse, output_shape=std_size)  # warp
    warped = warped * 255  # note output from wrap is double image (value range [0,1])
    warped = warped.astype('uint8')
    return warped, tform


def write_video(rois, target_path, write_fps):
    # fps = 25
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    frame_width, frame_height = rois.shape[1], rois.shape[2]
    out = cv2.VideoWriter(target_path, fourcc, write_fps, (frame_width, frame_height))
    frame_num = rois.shape[0]
    for index in range(frame_num):
        out.write(rois[index])
    out.release()


def get_frame_count(filename):
    cap = cv2.VideoCapture(filename)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    return total


def read_video(filename):
    cap = cv2.VideoCapture(filename)
    while (cap.isOpened()):
        ret, frame = cap.read()  # BGR
        if ret:
            yield frame
        else:
            break
    cap.release()


def crop_patch(video_pathname, landmarks, mean_face_landmarks, stablePntsIDs, STD_SIZE, window_margin, start_idx, stop_idx, crop_height, crop_width):
    """Crop mouth patch
    :param str video_pathname: pathname for the video_dieo
    :param list landmarks: interpolated landmarks
    """

    frame_idx = 0
    num_frames = get_frame_count(video_pathname)
    frame_gen = read_video(video_pathname)
    margin = min(num_frames, window_margin)
    while True:
        try:
            frame = frame_gen.__next__()  ## -- BGR
        except StopIteration:
            break
        if frame_idx == 0:
            q_frame, q_landmarks = deque(), deque()
            sequence = []

        q_landmarks.append(landmarks[frame_idx])
        q_frame.append(frame)
        if len(q_frame) == margin:
            smoothed_landmarks = np.mean(q_landmarks, axis=0)
            cur_landmarks = q_landmarks.popleft()
            cur_frame = q_frame.popleft()
            # -- affine transformation
            trans_frame, trans = warp_img(smoothed_landmarks[stablePntsIDs, :],
                                          mean_face_landmarks[stablePntsIDs, :],
                                          cur_frame,
                                          STD_SIZE)
            trans_landmarks = trans(cur_landmarks)
            # -- crop mouth patch
            sequence.append(cut_patch(trans_frame,
                                      trans_landmarks[start_idx:stop_idx],
                                      crop_height // 2,
                                      crop_width // 2, ))
        if frame_idx == len(landmarks) - 1:
            while q_frame:
                cur_frame = q_frame.popleft()
                # -- transform frame
                trans_frame = apply_transform(trans, cur_frame, STD_SIZE)
                # -- transform landmarks
                trans_landmarks = trans(q_landmarks.popleft())
                # -- crop mouth patch
                sequence.append(cut_patch(trans_frame,
                                          trans_landmarks[start_idx:stop_idx],
                                          crop_height // 2,
                                          crop_width // 2, ))
            return np.array(sequence)
        frame_idx += 1
    return None


def detect_landmark(image, detector, predictor):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    rects = detector(gray, 1)
    coords = None
    for (_, rect) in enumerate(rects):
        shape = predictor(gray, rect)
        coords = np.zeros((68, 2), dtype=np.int32)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
    return coords


def read_fps(input_path):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()
    return fps


def preprocess_video(input_video_dir, video_filename, output_video_dir, face_predictor_path, mean_face_path):
    if not os.path.exists(os.path.join(output_video_dir, video_filename[:-4] + '_roi.mp4')):
        os.makedirs(output_video_dir, exist_ok=True)
    else:
        return True

    input_path = os.path.join(input_video_dir, video_filename)
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(face_predictor_path)
    mean_face_landmarks = np.load(mean_face_path)
    try:
        videogen = skvideo.io.vread(input_path)
    except:
        print(f"Failed to read video: {input_path}")
        return False

    frames = np.array([frame for frame in videogen])
    landmarks = [detect_landmark(frame, detector, predictor) for frame in frames]
    preprocessed_landmarks = landmarks_interpolate(landmarks)

    try:
        rois = crop_patch(input_path, preprocessed_landmarks, mean_face_landmarks, STABLE_PNTS_IDS, STD_SIZE,
                          window_margin=12, start_idx=48, stop_idx=68, crop_height=96, crop_width=96)
    except:
        print(f"Failed to preprocess video: {input_path}; passing whole video")
        rois = frames[..., ::-1]

    roi_path = os.path.join(output_video_dir, video_filename[:-4] + '_roi.mp4')
    # audio_fn = os.path.join(output_video_dir, video_filename[:-4] + '.wav')
    base_dir = os.path.dirname(roi_path)
    if not os.path.exists(base_dir):
        os.mkdir(base_dir)
    write_fps = read_fps(input_path)
    write_video(rois, roi_path, write_fps)

    # subprocess.run([
    #     "ffmpeg",
    #     "-i", input_path,
    #     "-f", "wav",
    #     "-vn",
    #     "-y", audio_fn,
    #     "-loglevel", "quiet"
    # ])
    return True


# def av_process1(metadata, file_dir, save_dir, max_workers):
#     file_open = open(metadata, mode='r', encoding='utf-8')
#     json_data = list(json.load(file_open))
#     selected_videos = []
#     for index in tqdm(range(len(json_data))):
#         file_post_name = json_data[index]['file']
#         video_path = os.path.join(file_dir, file_post_name)
#         if os.path.isfile(video_path):
#             video_save_path = os.path.join(save_dir, file_post_name)
#             input_dir = os.path.dirname(video_path)
#             filename = os.path.basename(video_path)
#             output_dir = os.path.dirname(video_save_path)
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             selected_videos.append((input_dir, filename, output_dir))
#
#     print(len(selected_videos))
#
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(
#                 preprocess_video,
#                 input_dir,
#                 filename,
#                 output_dir,
#                 FACE_PREDICTOR_PATH,
#                 MEAN_FACE_PATH
#             ): (input_dir, filename)
#             for input_dir, filename, output_dir in selected_videos
#         }
#
#         for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing ..."):
#             input_dir, filename = futures[future]
#             try:
#                 result = future.result()
#                 if not result:
#                     print(f"[WARN] Failed to process video: {os.path.join(input_dir, filename)}")
#             except Exception as e:
#                 print(f"[ERROR] Error in video {os.path.join(input_dir, filename)}: {e}")


# def av_process2(metadata_dir, file_dir, save_dir1, max_workers):
#     json_paths = []
#     for dirpath, dirnames, filenames in os.walk(metadata_dir):
#         for filename in filenames:
#             if filename.lower().endswith('.json'):
#                 file_path = os.path.abspath(os.path.join(dirpath, filename))
#                 json_paths.append(file_path)
#     selected_videos = []
#     for json_path in tqdm(json_paths):
#         video_path = str(json_path).replace('.json', '.mp4').replace('meta', '')
#         if os.path.exists(video_path):
#             video_save_path = video_path.replace(file_dir, save_dir1)
#             input_dir = os.path.dirname(video_path)
#             filename = os.path.basename(video_path)
#             output_dir = os.path.dirname(video_save_path)
#             if not os.path.exists(output_dir):
#                 os.makedirs(output_dir)
#             selected_videos.append((input_dir, filename, output_dir))
#
#     print(len(selected_videos))
#
#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(
#                 preprocess_video,
#                 input_dir,
#                 filename,
#                 output_dir,
#                 FACE_PREDICTOR_PATH,
#                 MEAN_FACE_PATH
#             ): (input_dir, filename)
#             for input_dir, filename, output_dir in selected_videos
#         }
#
#         for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing ..."):
#             input_dir, filename = futures[future]
#             try:
#                 result = future.result()
#                 if not result:
#                     print(f"[WARN] Failed to process video: {os.path.join(input_dir, filename)}")
#             except Exception as e:
#                 print(f"[ERROR] Error in video {os.path.join(input_dir, filename)}: {e}")

def av_process3(metadata, file_dir, save_dir1, max_workers):
    file_open = open(metadata, mode='r', encoding='utf-8')
    reader = list(json.load(file_open))
    selected_videos = []
    for row in tqdm(reader):
        video_path = os.path.join(file_dir, row['file'])
        if os.path.isfile(video_path):
            video_save_path = os.path.join(save_dir1, row['file'])
            input_dir = os.path.dirname(video_path)
            filename = os.path.basename(video_path)
            output_dir = os.path.dirname(video_save_path)
            os.makedirs(output_dir, exist_ok=True)
            selected_videos.append((input_dir, filename, output_dir))

    print(len(selected_videos))

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                preprocess_video,
                input_dir,
                filename,
                output_dir,
                FACE_PREDICTOR_PATH,
                MEAN_FACE_PATH
            ): (input_dir, filename)
            for input_dir, filename, output_dir in selected_videos
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Processing ..."):
            input_dir, filename = futures[future]
            try:
                result = future.result()
                if not result:
                    print(f"[WARN] Failed to process video: {os.path.join(input_dir, filename)}")
            except Exception as e:
                print(f"[ERROR] Error in video {os.path.join(input_dir, filename)}: {e}")


if __name__ == "__main__":
    # metadata = '/data/ljc/deepfake/dataset/LAV-DF/metadata.min.json'
    # file_dir = '/data/ljc/deepfake/dataset/LAV-DF/'
    # save_dir = '/data2/ljc/TDL/LAV-DF-Preprocess/'

    # metadata = '/data/ljc/deepfake/dataset/AV-Deepfake1M/val_metadata.json'
    # file_dir = '/data/ljc/deepfake/dataset/AV-Deepfake1M/val/'
    # save_dir = '/data2/ljc/TDL/AV-Deepfake1M-Preprocess/val/'

    # metadata = '/data/ljc/deepfake/dataset/AV-Deepfake1M-PlusPlus/val_metadata.json'
    # file_dir = '/data/ljc/deepfake/dataset/AV-Deepfake1M-PlusPlus/val/'
    # save_dir = '/data2/ljc/TDL/AV-Deepfake1M-PlusPlus-Preprocess/val/'

    # metadata = '/data/ljc/deepfake/dataset/AV-Deepfake1M/train_metadata.json'
    # file_dir = '/data/ljc/deepfake/dataset/AV-Deepfake1M/train/'
    # save_dir = '/data2/ljc/TDL/AV-Deepfake1M-Preprocess/train/'

    metadata = '/data/ljc/deepfake/dataset/AV-Deepfake1M-PlusPlus/train_metadata.json'
    file_dir = '/data/ljc/deepfake/dataset/AV-Deepfake1M-PlusPlus/train/'
    save_dir = '/data2/ljc/TDL/AV-Deepfake1M-PlusPlus-Preprocess/train/'

    # metadata = '/data/ljc/deepfake/dataset/AV-Deepfake1M-PlusPlus/val_metadata.json'
    # file_dir = '/data/ljc/deepfake/dataset/AV-Deepfake1M-PlusPlus/val/'
    # save_dir = '/data2/ljc/TDL/AV-Deepfake1M-PlusPlus-Preprocess/val/'
    av_process3(metadata, file_dir, save_dir, 20)
