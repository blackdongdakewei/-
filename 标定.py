import cv2
import mediapipe as mp
import pyautogui
import numpy as np
import time
import pickle
from screeninfo import get_monitors
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

screen = get_monitors()[0]
SCREEN_WIDTH, SCREEN_HEIGHT = screen.width, screen.height

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)

LEFT_EYE_IDs = [33, 133]
RIGHT_EYE_IDs = [362, 263]

def get_eye_center(landmarks, ids, w, h):
    pts = [landmarks[i] for i in ids]
    xs = [p.x for p in pts]
    ys = [p.y for p in pts]
    return int(np.mean(xs) * w), int(np.mean(ys) * h)

# 3x3 标定点坐标（屏幕坐标）
GRID_POS = [(int(SCREEN_WIDTH * x), int(SCREEN_HEIGHT * y)) for y in [0.2, 0.5, 0.8] for x in [0.2, 0.5, 0.8]]

eye_points = []
screen_points = []

cap = cv2.VideoCapture(0)
w, h = int(cap.get(3)), int(cap.get(4))

print("开始眼动追踪标定，请根据提示注视屏幕红点...")
time.sleep(2)

for idx, (sx, sy) in enumerate(GRID_POS):
    print(f"请注视位置 {idx+1}/9: ({sx}, {sy})")
    pyautogui.moveTo(sx, sy)  # 显示鼠标作为注视点提示
    time.sleep(2.0)

    samples = []
    for _ in range(15):
        ret, frame = cap.read()
        if not ret:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if result.multi_face_landmarks:
            landmarks = result.multi_face_landmarks[0].landmark
            le = get_eye_center(landmarks, LEFT_EYE_IDs, w, h)
            re = get_eye_center(landmarks, RIGHT_EYE_IDs, w, h)
            eye_center = ((le[0] + re[0]) // 2, (le[1] + re[1]) // 2)
            samples.append(eye_center)

        cv2.imshow("Calibrating...", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    avg_eye = np.mean(samples, axis=0)
    eye_points.append(avg_eye)
    screen_points.append([sx, sy])
    print(f"已记录第 {idx+1} 个点")

cap.release()
cv2.destroyAllWindows()

# 拟合多项式映射
eye_points = np.array(eye_points)
screen_points = np.array(screen_points)

model_x = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(eye_points, screen_points[:, 0])
model_y = make_pipeline(PolynomialFeatures(2), LinearRegression()).fit(eye_points, screen_points[:, 1])

with open("model.pkl", "wb") as f:
    pickle.dump((model_x, model_y), f)

print("✅ 标定完成，模型已保存为 model.pkl")
