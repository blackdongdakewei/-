# === 依赖包 ===
from kokoro import KPipeline
from funasr import AutoModel
import pyaudio
import numpy as np
import requests
import json
import threading
import queue
import sounddevice as sd
import time
import cv2
import mediapipe as mp
import pickle
from screeninfo import get_monitors
from PIL import ImageGrab
from paddleocr import PaddleOCR
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QShortcut
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QPainter, QColor, QKeySequence
import sys
import keyboard

# === 配置参数 ===
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "deepseek-r1:1.5b"
AUDIO_FORMAT = pyaudio.paInt16
SAMPLE_RATE = 16000
CHUNK_SIZE = 9600
SILENT_LIMIT = 5

# === 初始化语音合成 ===
tts_pipeline = KPipeline(lang_code='z')
response_queue = queue.Queue()
stop_signal = threading.Event()

# === 初始化语音识别 ===
asr_model = AutoModel(model="paraformer-zh-streaming")
audio_interface = pyaudio.PyAudio()
input_stream = audio_interface.open(
    format=AUDIO_FORMAT,
    channels=1,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=CHUNK_SIZE
)

# === TTS 播放线程 ===
def tts_worker():
    while True:
        text = response_queue.get()
        if text is None:
            break
        stop_signal.clear()
        generator = tts_pipeline(text, voice='af_heart', speed=1.1, split_pattern=r'[。！？]')

        for _, _, audio in generator:
            if stop_signal.is_set():
                print("🔇 播放中断")
                sd.stop()  # 🛑 强制停止当前播放音频
                break
            sd.play(audio, samplerate=24000, blocking=False)
            start_time = time.time()
            while sd.get_stream().active:
                if stop_signal.is_set():
                    print("🔇 播放中断")
                    sd.stop()
                    break
                time.sleep(0.05)  # 避免过度占用CPU


# === 调用 LLM 回答 ===
def query_llm(prompt):
    try:
        system_prompt = """你是一个简洁中文助手：
1. 即使语句不完整也要理解意图
2. 回答口语化但专业
3. 最多3句
4. 优先使用本地知识（更新至2023年10月）
5. 技术问题需分步说明"""
        full_prompt = f"[INST] <<SYS>>\n{system_prompt}\n<</SYS>>\n\n{prompt} [/INST]"
        data = {
            "model": MODEL_NAME,
            "prompt": full_prompt,
            "stream": False
        }
        response = requests.post(OLLAMA_URL, json=data, timeout=30)
        if response.ok:
            result = response.json()
            response_queue.put(result['response'])
    except Exception as e:
        print(f"模型调用失败: {str(e)}")

# === OCR 模型和注视预测模型 ===
ocr_engine = PaddleOCR(
    use_angle_cls=True,
    lang='ch',
    det_model_dir='C:/ocrbao/ch_PP-OCRv4_det_infer',
    rec_model_dir='C:/ocrbao/ch_PP-OCRv4_rec_infer',
    cls_model_dir='C:/ocrbao/ch_ppocr_mobile_v2.0_cls_infer'
)
screen = get_monitors()[0]
SCREEN_WIDTH, SCREEN_HEIGHT = screen.width, screen.height
with open("model.pkl", "rb") as f:
    model_x, model_y = pickle.load(f)

# === MediaPipe 注视跟踪 ===
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
LEFT_EYE_IDs = [33, 133]
RIGHT_EYE_IDs = [362, 263]
gaze_point = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]

def get_eye_center(landmarks, ids, w, h):
    pts = [landmarks[i] for i in ids]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return int(np.mean(xs) * w), int(np.mean(ys) * h)

def gaze_tracker():
    global gaze_point
    cap = cv2.VideoCapture(0)
    w, h = int(cap.get(3)), int(cap.get(4))
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)
        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            le = get_eye_center(landmarks, LEFT_EYE_IDs, w, h)
            re = get_eye_center(landmarks, RIGHT_EYE_IDs, w, h)
            eye_center = np.array([(le[0] + re[0]) / 2, (le[1] + re[1]) / 2]).reshape(1, -1)
            sx = int(np.clip(model_x.predict(eye_center)[0], 0, SCREEN_WIDTH))
            sy = int(np.clip(model_y.predict(eye_center)[0], 0, SCREEN_HEIGHT))
            gaze_point = [sx, sy]
    cap.release()

# === PyQt 屏幕注视框 ===
class GazeOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 180)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(30)
        self.smooth_x = gaze_point[0]
        self.smooth_y = gaze_point[1]
        self.label = QLabel('', self)
        self.label.move(0, 185)
        self.label.setStyleSheet("color: white; background: rgba(0,0,0,180); padding: 4px;")

        # 快捷键：Ctrl+Q 触发 OCR 提问
        self.shortcut = QShortcut(QKeySequence("ctrl+Q"), self)
        self.shortcut.activated.connect(self.run_ocr)

        # 快捷键：Ctrl+W 停止语音播报
        self.shortcut_stop = QShortcut(QKeySequence("ctrl+W"), self)
        self.shortcut_stop.activated.connect(self.stop_tts)

    def stop_tts(self):
        stop_signal.set()
        print("🛑 快捷键触发：语音播报已中止")
        self.label.setText("语音已停止")

    def update_position(self):
        alpha = 0.2
        self.smooth_x = (1 - alpha) * self.smooth_x + alpha * (gaze_point[0] - self.width() // 2)
        self.smooth_y = (1 - alpha) * self.smooth_y + alpha * (gaze_point[1] - self.height() // 2)
        self.move(int(self.smooth_x), int(self.smooth_y))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(255, 0, 0, 100))
        painter.setPen(Qt.red)
        painter.drawRect(0, 0, self.width(), self.height())

    def run_ocr(self):
        x, y, w, h = int(self.smooth_x), int(self.smooth_y), self.width(), self.height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        result = ocr_engine.ocr(np.array(img), cls=True)
        text = ' '.join([line[1][0] for line in result[0]])
        if text:
            question = f"{text} 这是什么？"
            threading.Thread(target=query_llm, args=(question,), daemon=True).start()
            self.label.setText("提问中...")
        else:
            self.label.setText("无文字可识别")

    def get_ocr_text(self):
        x, y, w, h = int(self.smooth_x), int(self.smooth_y), self.width(), self.height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        result = ocr_engine.ocr(np.array(img), cls=True)
        text = ' '.join([line[1][0] for line in result[0]])
        return text.strip()


# === 主程序：语音识别循环 ===
QUESTION_KEYWORDS = ['废物','这是',"什么", "干嘛", "用途", "作用", "这是", "用来", "有用", "怎么用", "能干啥", "做什么", "有啥", "哪方面", "意义"]

def asr_loop():
    current_text = ""
    silent_count = 0
    asr_cache = {}
    print("🎤 语音助手已启动，直接讲话（Ctrl+C退出）...")

    while True:
        audio_data = input_stream.read(CHUNK_SIZE, exception_on_overflow=False)
        audio_array = np.frombuffer(audio_data, dtype=np.int16).astype('float32') / 32768
        result = asr_model.generate(input=audio_array, cache=asr_cache, is_final=False)

        if result and (text := result[0].get('text', '').strip()):
            current_text += text
            silent_count = 0
            print(f"\r识别中: {current_text}", end='', flush=True)

            # 停止指令
            # 🛑 播放中断关键词列表
            INTERRUPT_KEYWORDS = ["停", "停一下", "停止", "别说", "不说了", "闭嘴", "行了", "够了", "打住", "打断",
                                  "取消"]

            # 检测是否包含中断关键词
            if any(kw in current_text for kw in INTERRUPT_KEYWORDS):
                stop_signal.set()
                print("\n🛑 检测到中断指令，语音播报已中止")
                current_text = ""
                continue

            # 如果包含任何提问关键词，就触发提问
            if any(kw in current_text for kw in QUESTION_KEYWORDS):
                print("\n📤 检测到提问关键词，执行 OCR + 问答")
                user_question = current_text.strip()
                ocr_text = overlay.get_ocr_text()
                if ocr_text:
                    full_prompt = f"{user_question}\n\n以下是我看到的内容：\n{ocr_text}"
                    threading.Thread(target=query_llm, args=(full_prompt,), daemon=True).start()
                else:
                    print("⚠️ 无法从注视区域识别文字")
                current_text = ""
                continue

        else:
            silent_count += 1
            if silent_count >= SILENT_LIMIT and current_text:
                print("\n📤 静音超时，清除当前识别内容")
                current_text = ""
                silent_count = 0



# === 启动线程与 GUI ===
if __name__ == "__main__":
    threading.Thread(target=gaze_tracker, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()
    app = QApplication(sys.argv)
    overlay = GazeOverlay()
    overlay.show()
    threading.Thread(target=asr_loop, daemon=True).start()
    keyboard.add_hotkey('ctrl+q', overlay.run_ocr)
    keyboard.add_hotkey('ctrl+w', overlay.stop_tts)
    sys.exit(app.exec_())
