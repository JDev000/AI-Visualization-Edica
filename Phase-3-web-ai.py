from flask import Flask, Response
import cv2
import torch
import pyttsx3
import threading

app = Flask(__name__)

# Initialize the text-to-speech engine
engine = pyttsx3.init()
lock = threading.Lock()

# Load the pre-trained YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# List of target objects
target_objects = ['person', 'cell phone', 'biscuit', 'cup', 'bottle', 'knife', 'fork', 'spoon', 'bowl', 'dining table',
                  'scissors', 'keyboard', 'remote']

# Flag to keep track of detected objects
detected_objects = set()


def speak_object(object_name):
    with lock:
        engine.say(f"{object_name} detected")
        engine.runAndWait()


def generate_frames():
    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break

        # Run YOLOv5 model on the frame
        results = model(img)
        labels, cords = results.xyxyn[0][:, -1], results.xyxyn[0][:, :-1]

        current_detected = set()

        for i in range(len(labels)):
            row = cords[i]
            if row[4] >= 0.5:
                object_name = results.names[int(labels[i])]
                if object_name in target_objects:
                    x1, y1, x2, y2 = int(row[0] * 640), int(row[1] * 480), int(row[2] * 640), int(row[3] * 480)

                    # Color for bounding box
                    color = (0, 255, 0) if object_name == 'person' else (
                    255, 0, 0) if object_name == 'cell phone' else (0, 0, 255)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
                    cv2.putText(img, f'{object_name} {row[4]:.2f}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color,
                                2)

                    current_detected.add(object_name)

        # Check for new detections
        for obj in current_detected:
            if obj not in detected_objects:
                detected_objects.add(obj)
                threading.Thread(target=speak_object, args=(obj,)).start()

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return '''
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Object Detection</title>
    </head>
    <body>
        <h1>Object Detection</h1>
        <img src="/video_feed" width="640" height="480">
    </body>
    </html>
    '''


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True, threaded=True)
