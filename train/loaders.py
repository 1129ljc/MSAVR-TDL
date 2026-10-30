import os
import json
import random
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class AV_HuBERT_Feature(Dataset):
    def __init__(self, dataset_name, av_hubert_name, split, max_length):
        assert dataset_name in ['lavdf', 'avdeepfake1m', 'ddlav', 'avdeepfake1mpp'], print('dataset_name select in lavdf, avdeepfake1m, ddlav, avdeepfake1mpp')
        assert av_hubert_name in ['base_lrs3_iter4', 'large_vox_iter5', 'self_large_vox_433h'], print('av_hubert_ckpt select in base_lrs3_iter4, large_vox_iter5, self_large_vox_433h')
        assert split in ['train', 'val', 'test'], print('split select in train, val, test')
        assert max_length in [512, 640, 768, 896], print('max_length select in 512, 640, 768, 896')
        self.dataset_name = dataset_name
        self.av_hubert_name = av_hubert_name
        self.max_length = max_length

        file_open = open(os.path.join('/data2/ljc/TDL/av_hubert/', self.av_hubert_name, self.dataset_name,  self.split + '_metadata.json'), mode='r', encoding='utf-8')
        metadata = list(json.load(file_open))
        file_open.close()

        self.videos = []
        for index in tqdm(range(len(metadata))):
            item = dict(metadata[index])
            feature_path = item['feature_path']
            if os.path.isfile(feature_path):
                video_target = 1 if item['modify_video'] else 0
                audio_target = 1 if item['modify_video'] else 0
                fake_periods = item['fake_periods']
                self.videos.append([feature_path, video_target, audio_target, fake_periods])

    def __len__(self):
        return len(self.videos)

    def period2target(self, fake_segments):
        tfl_target = torch.zeros((self.max_length, 3))
        for fake_period in fake_segments:
            fake_period_indx = [int(x * 25) for x in fake_period]
            if fake_period_indx[0] < self.max_length - 1 and fake_period_indx[1] < self.max_length - 1:
                tfl_target[fake_period_indx[0]: fake_period_indx[1] + 1, 0] = 1
                for i in range(fake_period_indx[0], fake_period_indx[1] + 1):
                    tfl_target[i, 1] = i - fake_period_indx[0]
                    tfl_target[i, 2] = fake_period_indx[1] - i
        return tfl_target

    def __getitem__(self, idx):
        data_path, video_target, audio_target, fake_periods = self.videos[idx]
        data = np.load(data_path, allow_pickle=True)
        video_features = torch.tensor(data["visual"].astype(np.float16))
        audio_features = torch.tensor(data["audio"].astype(np.float16))
        t = min(video_features.shape[0], audio_features.shape[0], self.max_length)
        video_features = torch.concat([video_features[:t, :], torch.zeros([self.max_length - t, video_features.shape[1]])])
        audio_features = torch.concat([audio_features[:t, :], torch.zeros([self.max_length - t, audio_features.shape[1]])])
        tfl_target = self.period2target(fake_periods)
        return [video_features, audio_features, tfl_target, fake_periods]


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    batch_1 = [b[:3] for b in batch]
    batch_2 = [b[3] for b in batch]
    return torch.utils.data.dataloader.default_collate(batch_1), batch_2


def get_loaders(dataset_name, av_hubert_name, max_length, batch_size, workers, splits):
    g = torch.Generator()
    g.manual_seed(0)

    if splits is None:
        splits = ["train", "val", "test"]

    loaders = {}
    for split in splits:
        loaders[split] = DataLoader(
            AV_HuBERT_Feature(dataset_name, av_hubert_name, split, max_length),
            batch_size=batch_size,
            shuffle=True,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            collate_fn=collate_fn,
            pin_memory=True,
            drop_last=False,
        )
    return loaders

def adjust_data(data, device):
    # if task == "tfl":
    #     video_features, audio_features, labels = data[0]
    #     fake_periods = data[1]
    #     video_features, audio_features, labels = (
    #         video_features.to(device),
    #         audio_features.to(device),
    #         labels.float().to(device),
    #     )
    #     target = labels[:, :, 0].max(dim=-1)[0].cpu().numpy().tolist()
    #     return {
    #         "video_features": video_features,
    #         "audio_features": audio_features,
    #         "labels": labels,
    #         "fake_periods": fake_periods,
    #         "video_target": target,
    #         "audio_target": target,
    #     }
    # else:
    #     video_features, audio_features, video_target, audio_target = data
    #     video_features, audio_features = (video_features.to(device), audio_features.to(device))
    #     return {
    #         "video_features": video_features,
    #         "audio_features": audio_features,
    #         "video_target": video_target,
    #         "audio_target": audio_target,
    #     }

    video_features, audio_features, labels = data[0]
    fake_periods = data[1]
    video_features, audio_features, labels = (
        video_features.to(device),
        audio_features.to(device),
        labels.float().to(device),
    )
    target = labels[:, :, 0].max(dim=-1)[0].cpu().numpy().tolist()
    return {
        "video_features": video_features,
        "audio_features": audio_features,
        "labels": labels,
        "fake_periods": fake_periods,
        "video_target": target,
        "audio_target": target,
    }