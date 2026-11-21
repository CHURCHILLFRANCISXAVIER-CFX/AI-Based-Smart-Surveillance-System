import cv2
import time
import threading
import winsound
import os
from ultralytics import YOLO

# ================= CONFIG =================
RTSP_URL = "rtsp://admin:Pass%40123@192.168.29.18:554/Streaming/Channels/101"

SAVE_FOLDER = "captured"
ALARM_FILE = "alarm.wav"

PERSON_THRESHOLD = 1
ALERT_DELAY = 10
DRINK_PERSIST_FRAMES = 5

CONF_PERSON = 0.5
CONF_BOTTLE = 0.35
# ========================================

os.makedirs(SAVE_FOLDER, exist_ok=True)

print("🚀 Loading YOLO (CPU)")
model = YOLO("yolov8s.pt")

cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)

if not cap.isOpened():
    print("❌ Tiandy camera not accessible")
    exit()

alarm_running = False
last_alert_time = 0
drink_frames = 0

# ---------- ALARM ----------
def play_alarm():
    global alarm_running
    while alarm_running:
        winsound.PlaySound(ALARM_FILE, winsound.SND_FILENAME)

# ---------- HELPERS ----------
def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2)//2, (y1 + y2)//2)

print("🎥 Alcohol Nuisance Detection Started (Tiandy CCTV)")

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        time.sleep(0.2)
        continue

    results = model(frame, conf=0.3, verbose=False)

    persons = []
    drinks = []
    drinking_detected = False

    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])
            x1, y1, x2, y2 = map(int, b.xyxy[0])

            if cls == 0 and conf >= CONF_PERSON:
                persons.append((x1, y1, x2, y2))
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),2)

            elif cls in [39, 41, 44] and conf >= CONF_BOTTLE:
                drinks.append((x1, y1, x2, y2))
                cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)

    for (px1, py1, px2, py2) in persons:
        face_top = py1
        face_bottom = py1 + int((py2 - py1) * 0.35)
        face_left = px1 + int((px2 - px1) * 0.2)
        face_right = px2 - int((px2 - px1) * 0.2)

        cv2.rectangle(frame,
                      (face_left, face_top),
                      (face_right, face_bottom),
                      (255,255,0),1)

        for drink in drinks:
            dcx, dcy = box_center(drink)
            if face_left < dcx < face_right and face_top < dcy < face_bottom:
                drinking_detected = True
                cv2.putText(frame,"ALCOHOL NUISANCE",
                            (px1, py1-10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,(0,0,255),2)

    if drinking_detected:
        drink_frames += 1
    else:
        drink_frames = 0

    nuisance = drink_frames >= DRINK_PERSIST_FRAMES and len(persons) >= PERSON_THRESHOLD
    now = time.time()

    if nuisance and (now - last_alert_time > ALERT_DELAY):
        print("🚨 ALCOHOL NUISANCE DETECTED")
        cv2.imwrite(f"{SAVE_FOLDER}/evidence_{int(time.time())}.jpg", frame)

        if not alarm_running:
            alarm_running = True
            threading.Thread(target=play_alarm, daemon=True).start()

        last_alert_time = now

    if not nuisance:
        alarm_running = False
        winsound.PlaySound(None, winsound.SND_PURGE)

    status = "ALCOHOL" if nuisance else "NORMAL"
    cv2.putText(frame, f"STATUS: {status}",
                (10,30),
                cv2.FONT_HERSHEY_SIMPLEX,1,
                (0,0,255) if nuisance else (0,255,0),2)

    cv2.imshow("Alcohol Nuisance Detection (Tiandy)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

alarm_running = False
winsound.PlaySound(None, winsound.SND_PURGE)
cap.release()
cv2.destroyAllWindows()