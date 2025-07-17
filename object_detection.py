import cv2
from ultralytics import YOLO

# Load the model
model = YOLO("yolov8n.pt")

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run object detection
    results = model(frame, stream=True)

    # Plot results
    for r in results:
        annot_frame = r.plot()
        cv2.imshow("Real-Time Object Detection", annot_frame)

    # Exit key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
