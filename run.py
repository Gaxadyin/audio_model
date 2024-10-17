import logging
from microphone_read import MicrophoneReader
from predict import PhonemePredictor
from concurrent.futures import ThreadPoolExecutor

# 日志写出到文件
logging.basicConfig(level=logging.INFO, filename='log.txt', filemode='w', format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

executor = ThreadPoolExecutor(max_workers=1)

predicter = PhonemePredictor()
reader = MicrophoneReader(chuck_size=400, step_size=160, rate=16000)
import time
import os

audio_dir = "./tmp_audios"

# 若目录不存在，则创建目录
if not os.path.exists(audio_dir):
    os.makedirs(audio_dir)

# 清空目录下的所有文件
for file in os.listdir(audio_dir):
    os.remove(os.path.join(audio_dir, file))

def stream_predict():
    print("start predict")
    while True:
        try:
            audio_bytes = reader.get_next_bytes()
            # print(mfcc_ndarray)
            if audio_bytes is not None:
                # 保存临时文件，用于调试,文件名为当前时间戳
                timestamp = str(int(time.time()))
                audio_file = os.path.join(audio_dir, timestamp + ".wav")
                reader.save_bytes(audio_bytes, audio_file)
                # 批量预估音素
                print("predicting audio file: ", audio_file)
                logging.info("predicting audio file: " + audio_file)
                preds = predicter.predict_audio(audio_file)
                for pred in preds:
                    print(pred,end=' ')
                    # 将预估结果写入日志(log.txt)
                    # logging.info(pred)
            else:
                break
        except Exception as e:
            print(e)
            logging.error(e)
            
# 启动麦克风读取音频 
reader.start_reading_bytes()

# 等待约0.1秒，确保麦克风已经开始读取音频
time.sleep(0.1)
            
# 启动一个新线程，用于从麦克风读取音频并预测音素
executor.submit(stream_predict)