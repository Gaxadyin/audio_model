from predict import PhonemePredictor
import sounddevice as sd
import numpy as np
import time
import queue
import logging
# Parameters for the audio stream
duration = 10.0  # seconds
sample_rate = 16000  # Hz
channels = 1
predicter = PhonemePredictor()

logging.basicConfig(level=logging.INFO,filename='./log.txt')

# Function to list and select audio devices
def select_device(device_type):
    devices = sd.query_devices()
    print(f"\nAvailable {device_type} devices:")
    for i, device in enumerate(devices):
        if (device_type == "input" and device['max_input_channels'] > 0) or \
           (device_type == "output" and device['max_output_channels'] > 0):
            device_info = f"name: {device['name']}, hostapi: {device['hostapi']}, " \
                          f"max_input_channels: {device['max_input_channels']}, " \
                          f"max_output_channels: {device['max_output_channels']}"
            print(i, device_info)
    
    if len(devices) == 1:
        print("Only one device found, using default device")
        return 0
    
    while True:
        try:
            selection = int(input(f"Select {device_type} device number: "))
            if 0 <= selection < len(devices):
                return selection
            else:
                print("Invalid selection. Please try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

# Select input and output devices
input_device = select_device("input") 
output_device = select_device("output")


# Create a queue for audio data

# Callback function for the input stream
def input_callback(indata, frames, time, status):
    logging.info(f"Input callback: frames={frames}, time={time}, status={status}")
    # logging.info(f"Input callback: indata={indata[:50]}")
    indata = indata.reshape(-1)
    logging.info(f"indata: {indata.max()}")
    preds = predicter.predict(indata)
    print(preds)
    
    # audio_queue.put(indata)

# Callback function for the output stream
# def output_callback(outdata, frames, time, status):
#     if status:
#         print(status)
#     try:
#         data = audio_queue.get_nowait()
#         print(len(data))
#         outdata[:] = data
#     except queue.Empty:
#         outdata[:] = np.zeros((frames, channels))

# Create the input stream
input_stream = sd.InputStream(
    device=input_device,
    dtype='float32',
    samplerate=sample_rate,
    channels=channels,
    callback=input_callback
)

# Create the output stream
# output_stream = sd.RawOutputStream(
#     device=output_device,
#     dtype='int16',
#     samplerate=sample_rate,
#     channels=channels,
#     callback=output_callback
# )

# Start the streams
with input_stream:
    print(f"Recording and playing back for {duration} seconds...")
    sd.sleep(int(duration * 1000))



