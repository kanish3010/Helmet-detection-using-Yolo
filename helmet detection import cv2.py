import cv2
from ultralytics import YOLO  # Import YOLO from Ultralytics

# Load the YOLOv8 model
model = YOLO("yolov8n.pt")  # Use 'yolov8m.pt' for better accuracy

# Open webcam
cap = cv2.VideoCapture(0)  # Use 1 if you have an external webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run YOLO on the frame
    results = model(frame)

    # Draw detections on frame
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])

            # Check if detected object is a helmet (Class ID may vary)
            if cls == 0:  # Modify based on your trained model
                color = (0, 255, 0)  # Green for helmet
                label = f"Helmet {conf:.2f}"
            else:
                color = (0, 0, 255)  # Red for no helmet
                label = f"No Helmet {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Show the output
    cv2.imshow("Helmet Detection", frame)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()