# -*- coding: utf-8 -*-
"""
microphone reader ui
"""
import tkinter as tk
from microphone_read import MicrophoneReader
from predict import PhonemePredictor
from concurrent.futures import ThreadPoolExecutor
executor = ThreadPoolExecutor(max_workers=1)


predicter = PhonemePredictor()
reader = MicrophoneReader()


class MicrophoneUI(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("麦克风音素识别Demo")
        self.geometry("600x600")  # 增加高度为300，以便适应新增的控件
        
        # 增加一个使用说明标签
        self.instruction_label = tk.Label(self, text="请按下'开启麦克风'按钮开始录音，按下'停止麦克风'按钮将停止录音并识别音素。")
        self.instruction_label.pack(pady=(5, 0))  # 增加上方间距，让标签与按钮有适当的空间
        
        # 添加一个横线
        self.line = tk.Canvas(self, width=600, height=1, bg="black")
        self.line.pack()  # 增加上方间距，让横线与上方的标签有适当的空间
        
        # 上方的录音状态标签，居中显示
        self.recording_label = tk.Label(self, text="当前未录音")
        self.recording_label.pack(pady=(5, 0))  # 增加上方间距，让标签与按钮有适当的空间
        self.recording_label.pack(side=tk.TOP, expand=True)  # 添加expand属性可以让标签居中
        
        # 创建按钮框架并垂直排列按钮
        button_frame = tk.Frame(self)
        button_frame.pack(pady=10)  # 增加上方的间距

        self.start_button = tk.Button(button_frame, text="开启麦克风", command=self.start_recording)
        self.start_button.grid(row=0, column=0, padx=5)  # 在按钮框架中左侧摆放

        self.stop_button = tk.Button(button_frame, text="停止麦克风", command=self.stop_recording)
        self.stop_button.grid(row=0, column=1, padx=5)  # 按钮并排并留有间距
        
        # 增加一个清理文本按钮
        self.clear_button = tk.Button(button_frame, text="清空文本", command=self.clear_text)
        self.clear_button.grid(row=0, column=2, padx=5)
        
        
        # 添加"识别结果"标签
        self.result_label = tk.Label(self, text="识别结果")
        self.result_label.pack(pady=(5, 0))  # 增加间距让其与录音状态标签保持适当距离

        # 增加多行文本框，用于显示结果并设置为只读
        self.text = tk.Text(self, height=10, state=tk.DISABLED)  # 初始状态设置为只读
        self.text.pack(pady=(10, 20), fill=tk.BOTH, expand=True)  # 添加间距并使文本框扩展填满
        
        self.is_recording = False
    
        
    def start_recording(self):
        if self.is_recording:
            print("Already recording")
            return
        
        self.is_recording = True
        self.recording_label["text"] = "当前正在录音"
        reader.start_reading_bytes()
        
    
    def stop_recording(self):
        self.is_recording = False
        self.recording_label["text"] = "正在识别...请稍等"
        reader.close_bytes(save_path='tmp.wav')
        self.update_text()
        self.recording_label["text"] = "当前未录音"
        
    def update_text(self):
        results = predicter.predict_audio(audio_path="tmp.wav")
        results = ", ".join(results)
        print(results)
        # 设置文本内容，替换原有内容
        self.text.config(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.insert(tk.END, results)
        self.text.config(state=tk.DISABLED)
            
    def clear_text(self):
        # 设置文本内容为空
        self.text.config(state=tk.NORMAL)
        self.text.delete(1.0, tk.END)
        self.text.config(state=tk.DISABLED)


if __name__ == "__main__":
    app = MicrophoneUI()
    app.mainloop()