import os
import cv2
import json
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from python_speech_features import logfbank
from tqdm import tqdm

from utils import Compose, CenterCrop, Normalize
from fairseq import checkpoint_utils


def extract_features(model, video_path, audio_path, transform, device, video_frame_num):
    # print('video_path :', video_path)
    # print('audio_path :', audio_path)
    # print('video_frame_num :', video_frame_num)

    # read video
    video_cap = cv2.VideoCapture(video_path)
    frame_num = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    frame_last = None
    for index in range(frame_num):
        ret, frame = video_cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(frame)
            frame_last = frame.copy()
    if len(frames) < video_frame_num:
        buchong = video_frame_num - len(frames)
        for index in range(buchong):
            frames.append(frame_last)
    elif len(frames) > video_frame_num:
        frames = frames[0:video_frame_num]
    frames = np.stack(frames)
    frames = transform(frames)
    video_feats = torch.FloatTensor(frames).unsqueeze(0).unsqueeze(0).to(device)
    # print('video_feats.shape :', video_feats.shape)

    # read audio
    wav_data, sample_rate = librosa.load(audio_path, sr=16000, mono=True)
    assert sample_rate == 16000 and len(wav_data.shape) == 1
    audio_feats = logfbank(wav_data, samplerate=sample_rate).astype(np.float32)
    stack_order_audio = 4
    if len(audio_feats) % stack_order_audio != 0:
        pad = stack_order_audio - len(audio_feats) % stack_order_audio
        audio_feats = np.concatenate([audio_feats, np.zeros((pad, audio_feats.shape[1]), dtype=audio_feats.dtype)])
    audio_feats = audio_feats.reshape(-1, stack_order_audio * audio_feats.shape[1])
    if audio_feats.shape[0] < video_frame_num:
        audio_feats_pad = audio_feats[-1, :].reshape(1, -1)
        audio_feats_pad = np.repeat(audio_feats_pad, repeats=video_frame_num - audio_feats.shape[0], axis=0)
        audio_feats = np.concatenate([audio_feats, audio_feats_pad], axis=0)
    elif audio_feats.shape[0] > video_frame_num:
        audio_feats = audio_feats[0:video_frame_num, :]
    audio_feats = torch.from_numpy(audio_feats.astype(np.float32))
    with torch.no_grad():
        audio_feats = F.layer_norm(audio_feats, audio_feats.shape[1:])
    audio_feats = audio_feats.transpose(0, 1).unsqueeze(dim=0).to(device)
    # print('audio_feats.shape :', audio_feats.shape)

    # extract feature
    with torch.no_grad():
        f_audio, _ = model.extract_finetune({"video": None, "audio": audio_feats}, None, None)
        f_audio = f_audio.squeeze(0).cpu().numpy()
        # print('f_audio.shape :', f_audio.shape)
        f_video, _ = model.extract_finetune({"video": video_feats, "audio": None}, None, None)
        f_video = f_video.squeeze(0).cpu().numpy()
        # print('f_video.shape :', f_video.shape)
        # f_mm, _ = model.extract_finetune({"video": video_feats, "audio": audio_feats}, None, None)
        # f_mm = f_mm.squeeze(0).cpu().numpy()
    return f_audio, f_video


def get_file_paths(metadata, data_audio_path, data_video_path, save_path):
    file_paths = []
    txt_file_open = open(metadata, mode='r', encoding='utf-8')
    lines = txt_file_open.readlines()
    txt_file_open.close()
    for line in tqdm(lines):
        file_name, frame_num = str(line).strip().split(' ')
        mouth_roi_file_path = os.path.join(data_video_path, str(file_name).replace('.mp4', '_roi.mp4'))
        audio_file_path = os.path.join(data_audio_path, str(file_name).replace('.mp4', '.wav'))
        save_file_dir = os.path.dirname(os.path.join(save_path, file_name))
        file_paths.append([mouth_roi_file_path, audio_file_path, save_file_dir, int(frame_num)])
    return file_paths


def main(metadata, data_audio_path, data_video_path, save_path, ckpt_path, device):
    models, cfg, task = checkpoint_utils.load_model_ensemble_and_task([ckpt_path])
    model = models[0]
    if hasattr(models[0], 'decoder'):
        print(f"Checkpoint: fine-tuned")
        model = models[0].encoder.w2v_model
    else:
        print(f"Checkpoint: pre-trained w/o fine-tuning")
    model.to(device)
    model.eval()

    transform = Compose([
        Normalize(0.0, 255.0),
        CenterCrop((task.cfg.image_crop_size, task.cfg.image_crop_size)),
        Normalize(task.cfg.image_mean, task.cfg.image_std)
    ])

    file_paths = get_file_paths(metadata, data_audio_path, data_video_path, save_path)
    print('len(file_paths) :', len(file_paths))

    for file_path in tqdm(file_paths):
        mouth_roi_path = file_path[0]
        audio_path = file_path[1]
        save_file_dir = file_path[2]
        frame_num = file_path[3]

        try:
            feature_audio, feature_vid = extract_features(model, mouth_roi_path, audio_path, transform, device, frame_num)
        except:
            print(f"Unprocessed for file: {mouth_roi_path}")
            continue

        save_dict = {
            "visual": feature_vid.astype(np.float16),
            "audio": feature_audio.astype(np.float16)
        }
        file_name = os.path.basename(mouth_roi_path)
        save_path1 = os.path.join(save_file_dir, str(file_name)[:-4] + '.npz')
        os.makedirs(os.path.dirname(save_path1), exist_ok=True)
        np.savez(save_path1, **save_dict)


if __name__ == "__main__":
    metadata = '/mnt/119_data/dataset/video-deepfake-detection/AV-Deepfake1M-PlusPlus/test_metadata.txt'
    data_audio_path = '/mnt/119_data/dataset/video-deepfake-detection/AV-Deepfake1M-PlusPlus-Preprocess/val_audio/'
    data_video_path = '/mnt/119_data/dataset/video-deepfake-detection/AV-Deepfake1M-PlusPlus-Preprocess/val_video/'
    save_path = '/mnt/119_data/dataset/video-deepfake-detection/AV-HuBERT-Feature/base_lrs3_iter4/AVDeepFake1MPlusPlus/val/'
    ckpt_path = '/mnt/119_data/dataset/video-deepfake-detection/AV-HuBERT/ckpt/base_lrs3_iter4.pt'
    device = torch.device('cuda:0')
    main(metadata, data_audio_path, data_video_path, save_path, ckpt_path, device)