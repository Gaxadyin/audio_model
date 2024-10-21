import os
import librosa
import numpy as np
import torchaudio
from textgrid import TextGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 空标签
BLANK_LABEL = 'sil'


# 加载并解析TextGrid标注
def load_textgrid(textgrid_path):
    tg = TextGrid.fromFile(textgrid_path)
    phoneme_intervals = []
    for interval in tg[1]:  # 假设 TextGrid 文件的第一个层是音素层
        if interval.mark:  
            phoneme_intervals.append((interval.minTime, interval.maxTime, interval.mark))
        else:
            # 如果没有标注，使用空标签
            phoneme_intervals.append((interval.minTime, interval.maxTime, BLANK_LABEL))
    return phoneme_intervals



# 加载音频文件并提取音素对应的音频片段(每个片段20ms)
def load_audio_and_labels(audio_path, textgrid_path, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    phoneme_intervals = load_textgrid(textgrid_path)
    segments = []
    labels = []

    for (start, end, phoneme) in phoneme_intervals:
        start_sample = int(start * sr)
        end_sample = int(end * sr)
        audio_segment = audio[start_sample:end_sample]
        # 保证每个片段长度为20毫秒
        segment_length = int(0.02 * sr)  # 20毫秒
        if len(audio_segment) >= segment_length:
            # 切分为多个20毫秒片段
            for i in range(0, len(audio_segment) - segment_length + 1, segment_length):
                segment = audio_segment[i:i + segment_length]
                segments.append(segment)
                labels.append(phoneme)
    return segments, labels


def load_audio_segments(audio_path, sr=16000):
    audio, _ = librosa.load(audio_path, sr=sr)
    
    # 每20ms切分一次
    segment_length = int(0.02 * sr)  # 20毫秒
    segments = []
    for i in range(0, len(audio) - segment_length + 1, segment_length):
        segment = audio[i:i + segment_length]
        segments.append(segment)
    return segments




def load_audio_segments_en(audio_path, sr=16000):
    """
    英文音频的处理方式：每25ms切分一次，以10ms为步长，整个1s的音频可以切分为100个片段
    Args:
        audio_path (_type_): _description_
        sr (int, optional): _description_. Defaults to 16000.

    Returns:
        _type_: _description_
    """
    audio, _ = librosa.load(audio_path, sr=sr)
    
    segment_length = int(0.025 * sr)  # 25毫秒
    step_length = int(0.01 * sr)  # 10毫秒
    
    segments = []
    for i in range(0, len(audio) - segment_length + 1, step_length):
        segment = audio[i:i + segment_length]
        segments.append(segment)
        
    return segments
    

def get_audio_segments_en(audio_ndarray, sr=16000):
    """
    英文音频的处理方式：每25ms切分一次，以10ms为步长，整个1s的音频可以切分为100个片段
    Args:
        audio_path (_type_): _description_
        sr (int, optional): _description_. Defaults to 16000.

    Returns:
        _type_: _description_
    """
    segment_length = int(0.025 * sr)  # 25毫秒
    step_length = int(0.01 * sr)  # 10毫秒
    
    segments = []
    for i in range(0, len(audio_ndarray) - segment_length + 1, step_length):
        segment = audio_ndarray[i:i + segment_length]
        segments.append(segment)
        
    return segments



# 提取MFCC特征
def extract_mfcc_features(segments, sr=16000, n_mfcc=25):
    mfcc_features = []
    for segment in segments:
        mfcc = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=n_mfcc)
        mfcc = mfcc.T  # 转置，使时间步在前
        mfcc = mfcc.flatten()  # 展平成一维
        mfcc_features.append(mfcc)
    return np.array(mfcc_features)


# 提取melspectrogramtez
def extract_melspectrogram_features(segments, sr=16000, n_mels=128):
    melspectrogram_features = []
    for segment in segments:
        melspectrogram = librosa.feature.melspectrogram(y=segment, sr=sr, n_mels=n_mels)
        melspectrogram = melspectrogram.T  # 转置，使时间步在前
        melspectrogram = melspectrogram.flatten()  # 展平成一维
        melspectrogram_features.append(melspectrogram)
    return np.array(melspectrogram_features)


# 加载数据集并处理
def preprocess_dataset(dataset_path, sr=16000, n_mfcc=25):
    audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
    all_segments = []
    all_labels = []

    for audio_file in audio_files:
        audio_path = os.path.join(dataset_path, audio_file)
        textgrid_path = audio_path.replace('.wav', '.TextGrid')
        if not os.path.exists(textgrid_path):
            print(f"Warning: TextGrid file not found for {audio_file}")
            continue
        segments, labels = load_audio_and_labels(audio_path, textgrid_path, sr=sr)
        if segments:
            mfcc_features = extract_mfcc_features(segments, sr=sr, n_mfcc=n_mfcc)
            all_segments.append(mfcc_features)
            all_labels.extend(labels)

    if all_segments:
        X = np.vstack(all_segments)
        y = np.array(all_labels)
    else:
        X = np.array([])
        y = np.array([])

    return X, y


# 加载数据，以每个音频文件为单位保存样本
def preprocess_dataset_v2(dataset_path, sr=16000, n_mfcc=25):
    audio_files = [f for f in os.listdir(dataset_path) if f.endswith('.wav')]
    all_segments = {}
    all_labels = {}

    for audio_file in audio_files:
        audio_path = os.path.join(dataset_path, audio_file)
        textgrid_path = audio_path.replace('.wav', '.TextGrid')
        audio_name = audio_file.split('.')[0]
        if not os.path.exists(textgrid_path):
            print(f"Warning: TextGrid file not found for {audio_file}")
            continue
        segments, labels = load_audio_and_labels(audio_path, textgrid_path, sr=sr)
        if segments:
            mfcc_features = extract_mfcc_features(segments, sr=sr, n_mfcc=n_mfcc)
            all_segments[audio_name] = np.array(mfcc_features)
            all_labels[audio_name] = np.array(labels)

    return all_segments, all_labels



# 气流强度估计
def estimate_airflow_intensity(audio_segment):
    """
    Args:
        audio_segment (_type_): _description_

    Returns:
        _type_: _description_
    """
    energy = np.sum(np.square(audio_segment))
    intensity = energy / len(audio_segment)
    return intensity



# 标签编码
def encode_labels(y):
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    return y_encoded, le


if __name__ == "__main__":
    text_path = 'data/valid/BAC009S0002W0423.TextGrid'
    audio_path = 'data/valid/BAC009S0002W0423.wav'
    print(load_audio_and_labels(audio_path, text_path, sr=16000))
