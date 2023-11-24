import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np


class EmotionDataset(Dataset):
    def __init__(self, vision_dir, audio_dir, interoception_dir):
        self.vision_filenames = [os.path.join(vision_dir, f) for f in os.listdir(vision_dir) if f.endswith('.txt')]
        self.audio_filenames = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.txt')]
        self.interoception_filenames = [os.path.join(interoception_dir, f) for f in os.listdir(interoception_dir) if f.endswith('.txt')]
        self.vision_filenames.sort()
        self.audio_filenames.sort()
        self.interoception_filenames.sort()

    def __len__(self):
        return min(len(self.vision_filenames), len(self.audio_filenames), len(self.interoception_filenames))

    def __getitem__(self, idx):
        vision_path = self.vision_filenames[idx]
        audio_path = self.audio_filenames[idx]
        interoception_path = self.interoception_filenames[idx]
        vision_data = np.loadtxt(vision_path, delimiter=' ').astype(np.float32)
        audio_data = np.loadtxt(audio_path, delimiter=' ').astype(np.float32)
        interoception_data = np.loadtxt(interoception_path, delimiter=' ').astype(np.float32)
        return torch.from_numpy(vision_data), torch.from_numpy(audio_data), torch.from_numpy(interoception_data)

def get_modalities(batch_size, shuffle, agent, _vision=True, _audio=True, _interoception=True):
    base_dir = '/home/zhangzehang2/emotion/'
    vision_dir_male = os.path.join(base_dir, 'video-txt4/Male')
    audio_dir_male = os.path.join(base_dir, 'audio-txt4/Male')
    interoception_dir_male = os.path.join(base_dir, 'interoception-txt/Male')
    vision_dir_female = os.path.join(base_dir, 'video-txt4/Female')
    audio_dir_female = os.path.join(base_dir, 'audio-txt4/Female')
    interoception_dir_female = os.path.join(base_dir, 'interoception-txt/Female')

    dataset_male = EmotionDataset(vision_dir=vision_dir_male, audio_dir=audio_dir_male, interoception_dir=interoception_dir_male)
    dataset_female = EmotionDataset(vision_dir=vision_dir_female, audio_dir=audio_dir_female, interoception_dir=interoception_dir_female)

    data_loader_male = DataLoader(dataset_male, batch_size=batch_size, shuffle=shuffle)
    data_loader_female = DataLoader(dataset_female, batch_size=batch_size, shuffle=shuffle)

    return data_loader_male if agent == 'A' else data_loader_female

