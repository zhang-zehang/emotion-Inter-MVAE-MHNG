import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from mvae_poe import MultiVAE,train

import argparse


# 引数解析のためのパーサーを作成
parser = argparse.ArgumentParser(description='MVAE Training Script')

# 引数を追加
parser.add_argument('--latent_dim', type=int, default=8, help='latent dimension size')
parser.add_argument('--batch_size', type=int, default=10, help='batch size for training')
parser.add_argument('--variational_beta', type=float, default=1.0, help='Weight for the KL divergence term.')
parser.add_argument('--lambda_vision', type=float, default=1.0, help='Weight for the vision reconstruction loss.')
parser.add_argument('--lambda_audio', type=float, default=1.0, help='Weight for the audio reconstruction loss.')

# 引数を解析
args = parser.parse_args()

# データセットクラスの定義
class EmotionDataset(Dataset):
    def __init__(self, vision_dir, audio_dir):
        self.vision_filenames = [os.path.join(vision_dir, f) for f in os.listdir(vision_dir)]
        self.audio_filenames = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir)]
        self.vision_filenames.sort()
        self.audio_filenames.sort()

    def __len__(self):
        return min(len(self.vision_filenames), len(self.audio_filenames))

    def __getitem__(self, idx):
        vision_path = self.vision_filenames[idx]
        audio_path = self.audio_filenames[idx]
        vision_data = np.loadtxt(vision_path, delimiter=' ').astype(np.float32)
        audio_data = np.loadtxt(audio_path, delimiter=' ').astype(np.float32)
        return torch.from_numpy(vision_data), torch.from_numpy(audio_data)

# データセットのインスタンス化
vision_dir = '/home/zhangzehang2/emotion/video-txt4/Male'
audio_dir = '/home/zhangzehang2/emotion/audio-txt4/Male'
dataset = EmotionDataset(vision_dir=vision_dir, audio_dir=audio_dir)

# データローダーの作成
batch_size = 10  # バッチサイズ
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# モデルのインスタンス化
latent_dim = 8  # 8種類の感情を想定しているため、潜在次元は8とします
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MultiVAE(latent_dim=latent_dim, device=device).to(device)

# 最適化器の設定
optimizer = torch.optim.Adam(model.parameters(), lr=0.000001)

# トレーニングパラメータ
num_epochs = 100  

# トレーニングループ
for epoch in range(num_epochs):
    loss_avg, means, logvars = train(args, model, data_loader, optimizer, device=device)
    
    # 最初のエポックとその後の10エポックごとに損失を出力
    if (epoch % 10 == 0):
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss_avg:.4f}')