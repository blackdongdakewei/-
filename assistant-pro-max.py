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
from PyQt5.QtWidgets import QApplication, QWidget, QTextEdit, QShortcut,QPushButton
from PyQt5.QtCore import Qt, QTimer, pyqtSignal,QPoint
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

# === OCR 和注视预测模型 ===
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

# === 注视跟踪 ===
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

# === 回答展示窗口（滚动版） ===
class ResponseWindow(QWidget):
    update_text_signal = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.full_width = 400
        self.full_height = 180
        self.ball_size = 60
        self.expanded = False
        self.dragging = False
        self.mouse_press_pos = None
        self.saved_pos = QPoint(SCREEN_WIDTH - self.ball_size - 20, SCREEN_HEIGHT - self.ball_size - 40)

        self.idle_timer = QTimer()
        self.idle_timer.setInterval(10000)
        self.idle_timer.timeout.connect(self.collapse_to_ball)

        self.resize(self.ball_size, self.ball_size)
        self.move(self.saved_pos)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        self.text_edit.setStyleSheet("""
            QTextEdit {
                color: white;
                background: rgba(0, 0, 0, 200);
                padding: 6px;
                font-size: 16px;
                border: none;
            }
        """)
        self.text_edit.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOn)
        self.text_edit.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.text_edit.hide()

        # ❗添加关闭按钮
        self.close_btn = QPushButton("×", self)
        self.close_btn.setFixedSize(20, 20)
        self.close_btn.setStyleSheet("""
            QPushButton {
                color: white;
                background-color: transparent;
                border: none;
                font-size: 16px;
            }
            QPushButton:hover {
                color: red;
            }
        """)
        self.close_btn.clicked.connect(self.collapse_to_ball)
        self.close_btn.hide()

        self.update_text_signal.connect(self._update_text)

    def mousePressEvent(self, event):
        if not self.expanded:
            self.dragging = True
            self.mouse_press_pos = event.globalPos() - self.frameGeometry().topLeft()
            self.press_time = time.time()

    def mouseMoveEvent(self, event):
        if not self.expanded and self.dragging:
            self.move(event.globalPos() - self.mouse_press_pos)

    def mouseReleaseEvent(self, event):
        if not self.expanded and self.dragging:
            self.dragging = False
            release_time = time.time()
            if release_time - self.press_time < 0.3:
                self.saved_pos = self.pos()  # 👉 保存当前位置
                self.expand_window()

    def expand_window(self):
        self.resize(self.full_width, self.full_height)

        # 以悬浮球为“右上角”展开
        new_x = self.saved_pos.x() - self.full_width + self.ball_size
        new_y = self.saved_pos.y()
        self.move(new_x, new_y)

        self.text_edit.resize(self.full_width, self.full_height)
        self.text_edit.show()

        self.close_btn.move(self.full_width - 25, 5)
        self.close_btn.show()

        self.expanded = True
        self.idle_timer.start()

    def collapse_to_ball(self):
        self.text_edit.hide()
        self.close_btn.hide()
        self.resize(self.ball_size, self.ball_size)
        self.move(self.saved_pos)  # 👉 恢复上一次位置，不跳回初始位置
        self.expanded = False
        self.idle_timer.stop()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if not self.expanded:
            painter.setBrush(QColor(0, 0, 0, 180))
            painter.setPen(QColor(255, 255, 255, 200))
            painter.drawEllipse(0, 0, self.ball_size, self.ball_size)

    def display_text(self, text):
        self.update_text_signal.emit(text)

    def _update_text(self, text):
        self.text_edit.setPlainText(text)
        self.text_edit.moveCursor(self.text_edit.textCursor().End)
        self.expand_window()

# === 注视红框覆盖窗口 ===
class GazeOverlay(QWidget):
    def __init__(self):
        super().__init__()
        self.resize(400, 180)
        self.setWindowFlags(
            Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(30)
        self.smooth_x = gaze_point[0]
        self.smooth_y = gaze_point[1]
        self.shortcut = QShortcut(QKeySequence("ctrl+Q"), self)
        self.shortcut.activated.connect(self.run_ocr)
        self.shortcut_stop = QShortcut(QKeySequence("ctrl+W"), self)
        self.shortcut_stop.activated.connect(self.stop_tts)

    def stop_tts(self):
        stop_signal.set()
        print("🛑 快捷键触发：语音播报已中止")
        response_window.display_text("🔇 语音已停止")

    def update_position(self):
        alpha = 0.2
        self.smooth_x = (1 - alpha) * self.smooth_x + alpha * (gaze_point[0] - self.width() // 2)
        self.smooth_y = (1 - alpha) * self.smooth_y + alpha * (gaze_point[1] - self.height() // 2)
        self.move(int(self.smooth_x), int(self.smooth_y))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(255, 0, 0, 40))
        painter.setPen(QColor(255, 0, 0, 100))
        painter.drawRect(0, 0, self.width(), self.height())

    def run_ocr(self):
        x, y, w, h = int(self.smooth_x), int(self.smooth_y), self.width(), self.height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        result = ocr_engine.ocr(np.array(img), cls=True)
        text = ' '.join([line[1][0] for line in result[0]])
        if text:
            question = f"{text} 这是什么？"
            threading.Thread(target=query_llm, args=(question,), daemon=True).start()
            response_window.display_text("🧠 提问中...")
        else:
            response_window.display_text("⚠️ 无文字可识别")

    def get_ocr_text(self):
        x, y, w, h = int(self.smooth_x), int(self.smooth_y), self.width(), self.height()
        img = ImageGrab.grab(bbox=(x, y, x + w, y + h))
        result = ocr_engine.ocr(np.array(img), cls=True)
        text = ' '.join([line[1][0] for line in result[0]])
        return text.strip()

# === LLM 调用 ===
def query_llm(prompt):
    try:
        system_prompt = """你是一个简洁中文助手：\n1. 即使语句不完整也要理解意图\n2. 回答口语化但专业\n3. 最多3句\n4. 优先使用本地知识（更新至2023年10月）\n5. 技术问题需分步说明"""
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
        else:
            response_window.display_text("❌ 模型响应失败")
    except Exception as e:
        print(f"模型调用失败: {str(e)}")
        response_window.display_text("❌ 模型调用失败")

# === TTS 播放线程 ===
def tts_worker():
    while True:
        text = response_queue.get()
        if text is None:
            break
        stop_signal.clear()
        response_window.display_text(text)
        generator = tts_pipeline(text, voice='af_heart', speed=1.1, split_pattern=r'[。！？]')
        for _, _, audio in generator:
            if stop_signal.is_set():
                print("🔇 播放中断")
                sd.stop()
                break
            sd.play(audio, samplerate=24000, blocking=False)
            while sd.get_stream().active:
                if stop_signal.is_set():
                    print("🔇 播放中断")
                    sd.stop()
                    break
                time.sleep(0.05)

# === 语音识别循环 ===
QUESTION_KEYWORDS = ['废物','这是',"什么", "干嘛", "用途", "作用", "用来", "有用", "怎么用", "能干啥", "做什么", "有啥", "哪方面", "意义"]

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

            INTERRUPT_KEYWORDS = ["停", "停一下", "停止", "别说", "不说了", "闭嘴", "行了", "够了", "打住", "打断", "取消"]
            if any(kw in current_text for kw in INTERRUPT_KEYWORDS):
                stop_signal.set()
                print("\n🛑 检测到中断指令，语音播报已中止")
                response_window.display_text("🔇 语音中断")
                current_text = ""
                continue

            if any(kw in current_text for kw in QUESTION_KEYWORDS):
                print("\n📤 检测到提问关键词，执行 OCR + 问答")
                user_question = current_text.strip()
                ocr_text = overlay.get_ocr_text()
                if ocr_text:
                    full_prompt = f"{user_question}\n\n以下是我看到的内容：\n{ocr_text}"
                    threading.Thread(target=query_llm, args=(full_prompt,), daemon=True).start()
                    response_window.display_text("🧠 提问中...")
                else:
                    response_window.display_text("⚠️ 无法识别注视区域文字")
                current_text = ""
                continue
        else:
            silent_count += 1
            if silent_count >= SILENT_LIMIT and current_text:
                print("\n📤 静音超时，清除当前识别内容")
                current_text = ""
                silent_count = 0

# === 主程序入口 ===
if __name__ == "__main__":
    threading.Thread(target=gaze_tracker, daemon=True).start()
    threading.Thread(target=tts_worker, daemon=True).start()
    app = QApplication(sys.argv)
    overlay = GazeOverlay()
    overlay.show()
    response_window = ResponseWindow()
    response_window.show()
    threading.Thread(target=asr_loop, daemon=True).start()
    keyboard.add_hotkey('ctrl+q', overlay.run_ocr)
    keyboard.add_hotkey('q', overlay.stop_tts)
    sys.exit(app.exec_())
