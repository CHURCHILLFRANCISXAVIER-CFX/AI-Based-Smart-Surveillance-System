import cv2
from ultralytics import YOLO

print("🚀 Loading YOLO...")
model = YOLO("yolov8s.pt")

print("🎥 Starting Laptop Camera...")
cap = cv2.VideoCapture(0)

# Optional camera settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Laptop camera not accessible")
    exit()

print("✅ Alcohol Nuisance Detection Started")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Run YOLO
    results = model(frame, conf=0.3, verbose=False)

    persons = []
    drinks = []

    for r in results:
        for b in r.boxes:
            cls = int(b.cls[0])
            conf = float(b.conf[0])

            x1, y1, x2, y2 = map(int, b.xyxy[0])

            # Person
            if cls == 0 and conf >= 0.5:
                persons.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Person",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 0),
                            2)

            # Bottle / Cup / Wine Glass
            elif cls in [39, 41, 44] and conf >= 0.35:
                drinks.append((x1, y1, x2, y2))
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, "Bottle",
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 0, 255),
                            2)

    # Your nuisance detection logic goes here
    # (face region, bottle near mouth, persistence, alarm, Telegram, etc.)

    cv2.imshow("Laptop Camera - Alcohol Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()