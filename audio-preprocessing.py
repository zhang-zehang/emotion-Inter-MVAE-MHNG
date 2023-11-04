import os
import librosa
import numpy as np
from sklearn.preprocessing import StandardScaler

def extract_mfcc_features(audio_file, n_mfcc=20):
    y, sr = librosa.load(audio_file, sr=48000)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)
    return np.concatenate((mfccs, mfcc_delta, mfcc_delta2), axis=0)

def pad_features(features, max_frames=300):
    padded_features = np.zeros((features.shape[0], max_frames))
    frames_to_copy = min(max_frames, features.shape[1])
    padded_features[:, :frames_to_copy] = features[:, :frames_to_copy]
    return padded_features

# 全ての特徴量を抽出して結合
all_features = []
file_information = []
base_input_dir = "/home/zhangzehang2/emotion/audio"
for i in range(1, 25):
    actor_dir = os.path.join(base_input_dir, f"Actor_{str(i).zfill(2)}")
    for file_name in os.listdir(actor_dir):
        if file_name.endswith(".wav"):
            file_path = os.path.join(actor_dir, file_name)
            features = extract_mfcc_features(file_path)
            all_features.append(features.T)  # Transpose to align features
            file_information.append((i, file_name))

# 特徴量を結合してスケーラーにフィット
all_features = np.vstack(all_features)
scaler = StandardScaler().fit(all_features)

# 正規化された特徴量を保存
for (actor_number, file_name) in file_information:
    actor_dir = os.path.join(base_input_dir, f"Actor_{str(actor_number).zfill(2)}")
    file_path = os.path.join(actor_dir, file_name)
    features = extract_mfcc_features(file_path)
    normalized_features = scaler.transform(features.T).T  # Transpose to normalize and then transpose back
    padded_features = pad_features(normalized_features)

    # 保存ディレクトリの決定
    gender = "Male" if actor_number % 2 != 0 else "Female"
    output_dir = f"/home/zhangzehang2/emotion/audio-txt4/{gender}"
    os.makedirs(output_dir, exist_ok=True)  # ディレクトリがなければ作成
    output_file_path = os.path.join(output_dir, file_name.replace(".wav", ".txt"))
    np.savetxt(output_file_path, padded_features)