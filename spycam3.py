import cv2
from ultralytics import YOLO

print("🚀 Loading YOLO...")
model = YOLO("yolov8s.pt")

print("🎥 Starting Laptop Camera...")
cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

if not cap.isOpened():
    print("❌ Laptop camera not accessible")
    exit()

print("✅ Person + Bottle Detection Started")

while True:

    ret, frame = cap.read()

    if not ret:
        print("❌ Camera frame not received")
        break

    # YOLO detection
    results = model(frame, conf=0.30, verbose=False)

    for r in results:

        for b in r.boxes:

            cls = int(b.cls[0])
            conf = float(b.conf[0])

            x1, y1, x2, y2 = map(int, b.xyxy[0])

            # =====================================
            # PERSON
            # COCO CLASS 0 = PERSON
            # =====================================

            if cls == 0 and conf >= 0.40:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Person {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 0, 0),
                    2
                )

            # =====================================
            # BOTTLE
            # COCO CLASS 39 = BOTTLE
            # =====================================

            elif cls == 39 and conf >= 0.35:

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Water Bottle {conf:.2f}",
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

    cv2.imshow(
        "Person + Water Bottle Detection",
        frame
    )

    # ESC to exit
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()