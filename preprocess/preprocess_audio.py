import os
import subprocess
import json
from tqdm import tqdm


def av_process3(metadata, file_dir, save_dir):
    file_open = open(metadata, mode='r', encoding='utf-8')
    reader = list(json.load(file_open))
    file_open.close()
    for row in tqdm(reader):
        file_name = row['file']
        video_path = os.path.join(file_dir, file_name)
        audio_path = os.path.join(save_dir, file_name[:-4] + '.wav')
        if os.path.isfile(video_path):
            # audio_dir = os.path.dirname(audio_path)
            # os.makedirs(audio_dir, exist_ok=True)
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-f", "wav",
                "-vn",
                "-y", audio_path,
                "-loglevel", "quiet"
            ])
            # try:
            #     cmd = 'ffmpeg -i ' + str(video_path) + ' -f wav -vn -y ' + str(audio_path) + ' -loglevel quiet'
            #     os.system(cmd)
            # except:
            #     pass


def av_process4(metadata, file_dir, save_dir):
    file_open = open(metadata, mode='r', encoding='utf-8')
    lines = file_open.readlines()
    for line in tqdm(lines):
        file_name = str(line).strip()
        video_path = os.path.join(file_dir, file_name)
        audio_path = os.path.join(save_dir, file_name[:-4] + '.wav')
        if os.path.isfile(video_path):
            # subprocess.run([
            #     "ffmpeg",
            #     "-i", video_path,
            #     "-f", "wav",
            #     "-vn",
            #     "-y", audio_path,
            #     # "-loglevel", "quiet"
            # ])
            cmd = 'ffmpeg -i ' + str(video_path) + ' -f wav -vn -y ' + str(audio_path) + ' -loglevel quiet'
            os.system(cmd)


def av_process5(metadata, file_dir, save_dir):
    file_open = open(metadata, mode='r', encoding='utf-8')
    lines = file_open.readlines()
    file_open.close()
    for line in tqdm(lines):
        file_name = str(line).strip()
        video_path = os.path.join(file_dir, file_name)
        audio_path = os.path.join(save_dir, file_name[:-4] + '.wav')
        if os.path.isfile(video_path):
            subprocess.run([
                "ffmpeg",
                "-i", video_path,
                "-f", "wav",
                "-vn",
                "-y", audio_path,
                "-loglevel", "quiet"
            ])


if __name__ == "__main__":
    metadata = 'Z:/ljc/dataset/video-deepfake-detection/AV-Deepfake1M-PlusPlus/testA_files.txt'
    file_dir = 'Z:/ljc/dataset/video-deepfake-detection/AV-Deepfake1M-PlusPlus/testA/'
    save_dir = 'Z:/ljc/dataset/video-deepfake-detection/AV-Deepfake1M-PlusPlus-Preprocess/testA_audio/'
    av_process5(metadata, file_dir, save_dir)
