from predict import PhonemePredictor, load_audio_segments, extract_mfcc_features
from data_read import extract_melspectrogram_features
import numpy as np
predictor = PhonemePredictor()
# 这里修改为测试音频文件的路径
audio_file = "./test.wav"

results = predictor.predict_audio(audio_file, get_intensity=True)

print(np.array(results))