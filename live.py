from ultralytics import YOLO
import cv2

# Use the default camera (usually the built-in webcam)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Load a pre-trained YOLO model
model_path ='D:/Projects/Ear biometrics/Trained Colab model/Ear_Proj-20240530T233254Z-001/Ear_Proj/runs/detect/train2/weights/last.pt'
model = YOLO(model_path)  # load a custom model

threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from the video stream.")
        break

    results = model(frame)[0]

    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result

        if score > threshold:
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
            cv2.putText(frame, results.names[int(class_id)].upper(), (int(x1), int(y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 255, 0), 3, cv2.LINE_AA)

    cv2.imshow('Real-Time Object Detection', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
