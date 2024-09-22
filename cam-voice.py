import cv2
import torch
import pyttsx3

# Initialize the text-to-speech engine
engine = pyttsx3.init()

# Load the pre-trained YOLOv5 model from PyTorch hub (use 'yolov5s' for a small, fast model)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')  # 'yolov5s' is lightweight, fast

# List of target objects we want to detect (COCO dataset class names, including 'cell phone' and 'person')
target_objects = ['person', 'cell phone','biscuit', 'cup', 'bottle', 'knife', 'fork', 'spoon', 'bowl', 'dining table', 'scissors',
                  'keyboard', 'remote']

# Initialize video capture
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Set width
cap.set(4, 480)  # Set height

while True:
    success, img = cap.read()
    if not success:
        break

    # Run YOLOv5 model on the frame
    results = model(img)

    # Extract labels, bounding box coordinates, and confidence scores
    labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

    for i in range(len(labels)):
        row = cords[i]
        if row[4] >= 0.5:  # Only consider detections with confidence > 0.5
            object_name = results.names[int(labels[i])]
            if object_name in target_objects:  # Check if detected object is in the target list
                x1, y1, x2, y2 = int(row[0] * 640), int(row[1] * 480), int(row[2] * 640), int(row[3] * 480)

                # Assign a specific color for each object class
                if object_name == 'person':
                    color = (0, 255, 0)  # Green for person
                elif object_name == 'cell phone':
                    color = (255, 0, 0)  # Blue for phone
                else:
                    color = (0, 0, 255)  # Red for other objects

                # Draw a smaller bounding box (set thickness to 1 for smaller boxes)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)

                # Display object label and confidence score with larger text
                cv2.putText(img, f'{object_name} {row[4]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

                # Speak every time a target object is detected
                engine.say(f"{object_name} detected")
                engine.runAndWait()

    # Display the frame with bounding boxes
    cv2.imshow("Webcam", img)

    if cv2.waitKey(1) & 0xFF == ord("w"):
        break

cap.release()
cv2.destroyAllWindows()
