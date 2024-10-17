# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import librosa
from data_read import extract_mfcc_features, load_audio_segments_en as load_audio_segments
from typing import List, Dict


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LABELS_PATH = "./uniq_labels.txt"

def load_labels():
    with open(LABELS_PATH) as f:
        labels = f.readlines()
    return [label.strip() for label in labels]


class Network(torch.nn.Module):
    def __init__(self, label_size=113):
        super(Network, self).__init__()
        # TODO: Please try different architectures
        in_size = 13*(2*24+1)  # 637

        layers = [
            nn.Linear(in_size, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, label_size)
        ]
     
        self.laysers = nn.Sequential(*layers)


    def forward(self, A0):
        x = self.laysers(A0)
        return x
    
    
def process_audio_input(audio_segment: np.ndarray, sr=16000) -> np.ndarray:
    """
    处理音频输入(每个segment 20ms)，返回特征向量
    Args:
        audio_segment (np.ndarray): _description_
        sr (int, optional): _description_. Defaults to 16000.
        frame_length (int, optional): _description_. Defaults to 400.
        hop_length (int, optional): _description_. Defaults to 160.
    """
    mfcc_np = extract_mfcc_features(segments=[audio_segment], sr=sr, n_mfcc=13)
    return mfcc_np


# 气流强度估计
def estimate_airflow_intensity(audio_segment):
    """
    公式：能量/帧长
    能量=信号的平方和
    Returns:
        _type_: _description_
    """
    energy = np.sum(np.square(audio_segment))
    intensity = energy / len(audio_segment)
    return intensity


def estimate_airflow_intensity_advanced(audio_segment, sr=16000, frame_length=400, hop_length=160):
    """
    Args:
        audio_segment (_type_): _description_
        sr (int, optional): _description_. Defaults to 16000.
        frame_length (int, optional): _description_. Defaults to 400.
        hop_length (int, optional): _description_. Defaults to 160.

    Returns:
        _type_: _description_
    """
    # 计算短时能量
    energy = librosa.feature.rms(y=audio_segment, frame_length=frame_length, hop_length=hop_length)
    # 返回平均能量
    intensity = np.mean(energy)
    return intensity


class MfccItems(object):
    def __init__(self, X, context = 0):
        
        self.length  = X.shape[0]
        self.context = context

        if context == 0:
            self.X = X
        else:
            X = np.pad(X, ((context,context), (0,0)), 'constant', constant_values=(0,0))
            self.X = X

        
    def __len__(self):
        return self.length
        
    def __getitem__(self, i):
        if self.context == 0:
            xx = self.X[i].flatten()
        else:
            xx = self.X[i:(i + 2*self.context + 1)].flatten()
        return xx


class PhonemePredictor:
    def __init__(self, model_path='best_model.pt', context=24, labels=None):
        if labels is None:
            labels = load_labels()
        self.labels = labels
        self.model = Network(label_size=len(labels))
        self.model.load_state_dict(torch.load(model_path,weights_only=False, map_location=torch.device('cpu')))
        self.model.to(DEVICE)
        self.model.eval()
        print(f"Model weights loaded from {model_path}")
        self.context = context
        self.historical_mfcc = []
        # 用于记录历史音素标签，初始化为sil
        self.hist_labels = [0]
        
        # normalize相关参数       
        # label prob, 记录标签之间的转移概率，用于hmm解码
        # self.trans_prob = np.load(TRANS_PROB_PATH)
        
    def viterbi_decode(self, label_prob: np.ndarray) -> int:
        """
        使用viterbi解码，得到最可能的音素序列
        Args:
            label_prob (np.ndarray): 音素概率
        Returns:
            int: 音素索引
        """
        # TODO
        pass
    
    
    def simple_decode(self, label_prob: np.ndarray) -> int:
        """
        简单的解码器，返回当前概率最大的音素
        Args:
            label_prob (np.ndarray): _description_

        Returns:
            int: _description_
        """
        predicted_phoneme = np.argmax(label_prob)
        return predicted_phoneme
    
    
    def simple_decode_batch(self, label_prob: np.ndarray) -> np.ndarray:
        """
        Args:
            label_prob (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        predicted_phonemes = np.argmax(label_prob, axis=1)
        return predicted_phonemes
    
    
    def clear_hist(self):
        """
        清除历史信息
        """
        self.hist_labels = [0]
        self.historical_mfcc = []
        
    
    def get_intensity(self, audio_segment: np.ndarray) -> float:
        """
        估计音频片段的气流强度
        Args:
            audio_segment (np.ndarray): 音频片段
        Returns:
            float: 气流强度
        """
        return estimate_airflow_intensity(audio_segment)
    
    
    def predict_mfcc(self, mfcc_np: np.ndarray) -> int:
        """
        预测音素
        Args:
            mfcc_np (np.ndarray): mfcc特征
        Returns:
            int: 音素索引
        """
        # 首先计算音素概率
        label_prob = self.pred_mfcc_prob(mfcc_np)
        
        # 解码，得到最可能的音素序列
        predicted_phoneme = self.simple_decode(label_prob)
        return predicted_phoneme
    

    def pred_mfcc_prob(self, mfcc_np: np.ndarray) -> np.ndarray:
        """
        预测音素概率
        Args:
            mfcc_np (np.ndarray): mfcc特征
        Returns:
            np.ndarray: 音素概率
        """
        # 若mfcc_np不是二维的，则转换为二维
        if len(mfcc_np.shape) == 1:
            mfcc_np = mfcc_np.reshape(1, -1)
            
        # 对mfcc进行normalize
        mfcc_np = (mfcc_np - self.mean) / self.std
        
        self.historical_mfcc.append(mfcc_np)
        if len(self.historical_mfcc) > 2 * self.context + 1:
            self.historical_mfcc.pop(0)
        
        # 拼接历史mfcc, 若不足context, 则补0
        mfcc_concat = np.concatenate(self.historical_mfcc, axis=1)
        if mfcc_concat.shape[1] < 13*(2*self.context+1):
            mfcc_concat = np.pad(mfcc_concat, ((0, 0), (0, 13*(2*self.context+1)-mfcc_concat.shape[1])), 'constant')
        
        # mfcc_tensor = torch.Tensor(mfcc_concat).unsqueeze(0) # 不需要再增加维度
        mfcc_tensor = torch.Tensor(mfcc_concat)
        mfcc_tensor = mfcc_tensor.to(DEVICE)
        
        # print(mfcc_tensor.shape)
        
        # 推理
        with torch.no_grad():
            output = self.model(mfcc_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()
            
        # 返回音素索引
        return prob
    
    
    def predict_raw_mfcc_prob(self, mfcc: np.ndarray) -> np.ndarray:
        """
        不需要拼接历史mfcc，直接预测，数据在外部已经处理好
        Args:
            mfcc (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        # 若mfcc_np不是二维的，则转换为二维
        if len(mfcc.shape) == 1:
            mfcc = mfcc.reshape(1, -1)
        
        mfcc_tensor = torch.Tensor(mfcc)
        mfcc_tensor = mfcc_tensor.to(DEVICE)
        
        with torch.no_grad():
            output = self.model(mfcc_tensor)
            prob = torch.softmax(output, dim=1).cpu().numpy()
        
        return prob
    

    def predict(self, audio_segment: np.ndarray) -> int:
        """
        预测音素
        Args:
            audio_segment (np.ndarray): 音频片段
        Returns:
            int: 音素索引
        """
        mfcc_np = process_audio_input(audio_segment)
        return self.predict_mfcc(mfcc_np)
    
    
    def pred_prob(self, audio_segment: np.ndarray) -> np.ndarray:
        """
        预测音素概率
        Args:
            audio_segment (np.ndarray): 音频片段
        Returns:
            np.ndarray: 音素概率
        """
        mfcc_np = process_audio_input(audio_segment)
        return self.pred_mfcc_prob(mfcc_np)
     
    
    def predict_label(self, audio_segment: np.ndarray) -> str:
        """
        预测音素标签
        Args:
            audio_segment (np.ndarray): 音频片段
        Returns:
            str: 音素标签
        """
        predicted_phoneme = self.predict(audio_segment)
        return self.labels[predicted_phoneme]
    
    
    def predict_mfcc_label(self, mfcc_np: np.ndarray) -> str:
        """
        预测音素标签
        Args:
            audio_segment (np.ndarray): 音频片段
        Returns:
            str: 音素标签
        """
        predicted_phoneme = self.predict_mfcc(mfcc_np)
        return self.labels[predicted_phoneme]
    
    
    def predict_raw_mfcc(self, mfcc: np.ndarray) -> int:
        """
        预测音素标签
        Args:
            mfcc (np.ndarray): mfcc特征
        Returns:
            str: 音素标签
        """
        prob = self.predict_raw_mfcc_prob(mfcc)
        predicted_phoneme = self.simple_decode(prob)
        return predicted_phoneme
    
    
    def predict_raw_mfcc_label(self, mfcc: np.ndarray) -> str:
        """
        预测音素标签
        Args:
            mfcc (np.ndarray): mfcc特征
        Returns:
            str: 音素标签
        """
        pred_phoneme = self.predict_raw_mfcc(mfcc)
        return self.labels[pred_phoneme]
    
    
    def predict_audio(self, audio_path: str, sr=None, get_intensity=False) -> List[str]:
        """
        预测音频文件的音素标签
        Args:
            audio_path (str): 音频文件路径
            sr (int, optional): 采样率. Defaults to 16000.
        Returns:
            str: 音素标签
        """
        # 若采样率未指定，则默认为16000
        if sr is None:
            sr = 16000
        
        audio_segments = load_audio_segments(audio_path, sr)
        # 转换为mfcc np
        mfcc_np = extract_mfcc_features(segments=audio_segments, sr=sr, n_mfcc=13)
        predicted_phonemes = self.predict_mfcc_np(mfcc_np)
        
        # predicted_phonemes = [self.predict_label(segment) for segment in audio_segments]
        
        # results = []
        # # 获取气流强度
        # if get_intensity:
        #     intensity = [self.get_intensity(segment) for segment in audio_segments]
        #     results = [{"phoneme": p, "intensity": i, 'time': t*0.02} for t, p, i in zip(range(len(predicted_phonemes)), predicted_phonemes, intensity)]
        # else:
        #     results = [{"phoneme": p, 'time': t*0.02} for t, p in enumerate(predicted_phonemes)]
        
        return predicted_phonemes
    
    
    def predict_mfcc_np(self, mfcc_np: np.ndarray) -> List[str]:
        """
        预测mfcc np数组的音素标签
        Args:
            mfcc_np (np.ndarray): _description_

        Returns:
            List[str]: _description_
        """
        # normalize
        mfcc_np = (mfcc_np - mfcc_np.mean(axis=0)) / mfcc_np.std(axis=0)
        # mfcc_np = (mfcc_np - self.mean) / self.std
        
        mfcc_items = MfccItems(mfcc_np, context=self.context)
        
        labels = []
        for i in range(len(mfcc_items)):
            mfcc = mfcc_items[i]
            label = self.predict_raw_mfcc_label(mfcc)
            labels.append(label)
            
        return labels
    
    def predict_mfcc_file(self, mfcc_np_file: str) -> List[str]:
        """
        预测音频文件的音素标签
        Args:
            mfcc_np_file (str): mfcc特征文件路径
        Returns:
            str: 音素标签
        """
        mfcc_np = np.load(mfcc_np_file)
        
        return self.predict_mfcc_np(mfcc_np)
            