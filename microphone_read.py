# -*- coding: utf-8 -*-
"""
microphone reader    
"""
import pyaudio
import wave
import numpy as np
import time

from concurrent import futures
from concurrent.futures import ThreadPoolExecutor
from data_read import estimate_airflow_intensity

RATE = 16000
CHUNK = 320 # 每次读取20ms的数据
FORMAT = pyaudio.paFloat32
FORMAT_INT = pyaudio.paInt16

CHANNELS = 1

executor = ThreadPoolExecutor(max_workers=1)


from data_read import extract_mfcc_features


class MicrophoneReader(object):
    def __init__(self, chuck_size=320, rate=16000, step_size=320, window_size=24):
        self.pa = None
        self.is_close = True
        # 单次最多保留30s的音频数据
        self.max_size = rate * 60
        self.max_steps = self.max_size // step_size
        self.segments = np.zeros((step_size * window_size))
        self.audio_bytes = []
        self.index = 0
        self.mfcc_index = 0
        self.chuck_size = chuck_size
        self.rate = rate
        self.step_size = step_size
        self.window_size = window_size
        self.stream = None
        self.mfcc_num = 13
        self.mfcc_features = []
        # 临时存储mfcc特征
        self.mfcc_features_temp = []
        # 激活状态
        self.active = False
        # 激活阈值 1e-8
        self.threshold = 1e-8
        # 阈值2
        self.threshold2 = 200
        # 计数器
        self.counter = 0
        # tmp bytes
        self.tmp_bytes = b''
        # 平均气流强度
        self.average_airflow_intensity = 0
        self.seg_counter = 0
    
    
    def start_reading(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=FORMAT,
                                   channels=CHANNELS,
                                   rate=self.rate,
                                   input=True,
                                   frames_per_buffer=self.chuck_size)
        self.is_close = False
        
        # 开启线程读取音频数据
        executor.submit(self._read_audio)
        
    def start_reading_bytes(self):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=FORMAT_INT,
                                   channels=CHANNELS,
                                   rate=self.rate,
                                   input=True,
                                   frames_per_buffer=self.chuck_size)
        self.is_close = False
        
        # 开启线程读取音频数据
        executor.submit(self._read_bytes)
        
    
    def get_next_mfcc_ndarray_batch(self) -> np.ndarray:
        """
        Returns:
            np.ndarray: _description_
        """
        while len(self.mfcc_features) == 0 and not self.is_close:
            # 等待读取到足够的音频数据
            # print('waiting for enough data, ', len(self.mfcc_features), self.mfcc_index)
            time.sleep(0.1)
        
        if len(self.mfcc_features) > 0:
            mfcc_features = self.mfcc_features[0]
            self.mfcc_features = self.mfcc_features[1:]
            # 获取一组mfcc特征，不用进行归一化，predict时会进行归一化
            normalized_mfcc_features = np.array(mfcc_features)
            return normalized_mfcc_features
        
        return None
    
    
    def get_next_bytes(self) -> bytes:
        """
        Returns:
            bytes: _description_
        """
        while len(self.audio_bytes) == 0 and not self.is_close:
            # 等待读取到足够的音频数据
            # print('waiting for enough data, ', len(self.mfcc_features), self.mfcc_index)
            time.sleep(0.02)
        
        if len(self.audio_bytes) > 0:
            t_audio_bytes = self.audio_bytes[0]
            self.audio_bytes = self.audio_bytes[1:]
            return t_audio_bytes
        
        return None
        
    
    def get_next_mfcc_ndarray(self) -> np.ndarray:
        # 当数据达到最大steps时，删除最旧的50个step的数据(1s)
        if self.mfcc_index >= self.max_steps:
            self.mfcc_features = self.mfcc_features[50:]
            self.mfcc_index -= 50
            
        if len(self.mfcc_features) < self.mfcc_index + self.window_size + 1 and self.is_close:
            return None
        
        # 等待读取到足够的音频数据
        while len(self.mfcc_features) < self.mfcc_index + self.window_size + 1:
            # print('waiting for enough data, ', len(self.mfcc_features), self.mfcc_index)
            time.sleep(0.1)
        
        normalized_mfcc_features = np.array(self.mfcc_features)
        
        # 归一化mfcc特征
        normalized_mfcc_features = (normalized_mfcc_features - np.mean(normalized_mfcc_features, axis=0)) / np.std(normalized_mfcc_features, axis=0)
         
        # padding
        all_mfcc_features = np.pad(normalized_mfcc_features, ((self.window_size, 0), (0, 0)), 'constant', constant_values=0)
        
        # 截取最新的window_size个mfcc特征
        mfcc_features = all_mfcc_features[self.mfcc_index: self.mfcc_index + 2 *self.window_size + 1]
        
        
        # 更新mfcc_index
        self.mfcc_index += 1
        
        return mfcc_features.flatten()
    
    
    def _read_bytes(self):
        print('start reading ....')
        while not self.is_close:
            audio_data = self.stream.read(CHUNK)
            
            # 气流强度估计
            segment = np.frombuffer(audio_data, dtype=np.int16)
            
            airflow_intensity = estimate_airflow_intensity(segment)
            
            # print('airflow_intensity:', airflow_intensity)
            self.average_airflow_intensity += airflow_intensity
            self.seg_counter += 1
            
            if airflow_intensity > self.threshold2:
                if self.active == False:
                    print('已激活识别')
                    
                self.active = True
                self.counter = 0
            else:
                if self.active:
                    self.counter += 1
                    if self.counter > 15:
                        print('已停止识别')
                        self.active = False
                        # 根据平均气流强度和segment数量判断是否需要保存音频数据
                        ave_airflow_intensity = self.average_airflow_intensity/self.seg_counter
                        print('average_airflow_intensity:', ave_airflow_intensity)
                        print('seg_counter:', self.seg_counter)
                        
                        # 平均气流强度需要大于阈值才能保存音频数据
                        if ave_airflow_intensity > self.threshold2:
                            print('save audio bytes')
                            self.audio_bytes.append(self.tmp_bytes)
                        
                        self.tmp_bytes = b''
                        self.average_airflow_intensity = 0
                        self.seg_counter = 0
        
            if self.active:
                self.tmp_bytes += audio_data
        
        if len(self.tmp_bytes) > 0:
            self.audio_bytes.append(self.tmp_bytes)
            self.tmp_bytes = b''
        print('stop reading ....')
    
    
    def _read_audio(self):
        print('start reading ....')
        while not self.is_close: 
            if self.index % 100 == 0:
                print('read index:', self.index)
            
            audio_data = self.stream.read(CHUNK)
            audio_data = np.frombuffer(audio_data, dtype=np.float32)
            self.segments = np.append(self.segments, audio_data)
            
            while len(self.segments) > self.index * self.step_size + self.window_size:
                segment = self.segments[self.index * self.step_size: self.index * self.step_size + self.window_size]
                self.index += 1
                
                # 气流强度估计
                airflow_intensity = estimate_airflow_intensity(segment)
                # print('airflow_intensity:', airflow_intensity)
                
                # 若气流强度持续window_size个step都小于阈值，则将激活状态置为False，不再提取mfcc特征
                if airflow_intensity > self.threshold:
                    if self.active == False:
                        print('已激活识别')
                        # 若临时存储的mfcc特征不为空，则将其添加到mfcc_features中
                        if len(self.mfcc_features_temp) > 0:
                            self.mfcc_features.append(self.mfcc_features_temp)
                            self.mfcc_features_temp = []
                    
                    self.active = True
                    self.counter = 0
                else:
                    if self.active:
                        self.counter += 1
                        if self.counter > 10:
                            print('已停止识别')
                            self.active = False
                
                
                # 提取MFCC特征
                if self.active:
                    mfcc_features = extract_mfcc_features([segment], sr=RATE, n_mfcc=self.mfcc_num)
                    # 添加到临时存储的mfcc特征中
                    self.mfcc_features_temp.append(mfcc_features.reshape(-1))
            
            
            # 保证内存占用不会无限增加, 丢弃最旧的1s数据
            if len(self.segments) > self.max_size:
                self.segments = self.segments[self.rate:]
                self.index -= self.rate // self.step_size
        
        
        # 若临时存储的mfcc特征不为空，则将其添加到mfcc_features中
        if len(self.mfcc_features_temp) > 0:
            self.mfcc_features.append(self.mfcc_features_temp)
            self.mfcc_features_temp = []
        
        print('stop reading ....')
                

    def close(self):
        self.is_close = True
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        
    def close_bytes(self, save_path=None):
        self.is_close = True
        self.stream.stop_stream()
        self.stream.close()
        self.pa.terminate()
        
        # 保存音频数据
        if save_path:
            # 将所有bytes写入到文件中
            all_audio_bytes = b''.join(self.audio_bytes)
            
            wf = wave.open(save_path, 'wb')
            wf.setnchannels(CHANNELS)
            wf.setsampwidth(self.pa.get_sample_size(FORMAT_INT))
            wf.setframerate(RATE)
            wf.writeframes(all_audio_bytes)
            wf.close()
            self.audio_bytes = []
            
    
    def save_bytes(self, t_audio_bytes, save_path):
        """
        Args:
            t_audio_bytes (_type_): _description_
            save_path (_type_): _description_
        """
        wf = wave.open(save_path, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(self.pa.get_sample_size(FORMAT_INT))
        wf.setframerate(RATE)
        wf.writeframes(t_audio_bytes)
        wf.close()
        