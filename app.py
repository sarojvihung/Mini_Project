from flask import Flask, render_template, Response
import cv2
import torch
import numpy as np
from PIL import Image
from numpy import asarray, expand_dims
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from datetime import datetime
import pandas as pd
import pickle
import os
from keras_facenet import FaceNet
from model import initialize_model  

app = Flask(__name__)

# Load models and initialize components
HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
MyFaceNet = FaceNet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
spoofing_model = initialize_model(num_classes=2).to(device)
spoofing_model.load_state_dict(torch.load('anti_spoofing_model1.pth', map_location=device))
spoofing_model.eval()

# Load face database
with open("data1.pkl", "rb") as myfile:
    database = pickle.load(myfile)

attendance_file = "attendance.xlsx"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel(attendance_file, index=False)

# Image transformations for anti-spoofing model
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

def adjust_gamma(image, gamma):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)]).astype("uint8")
    return cv2.LUT(image, table)

def video_feed():
    cap = cv2.VideoCapture(0)
    real_count = 0
    fake_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = adjust_gamma(frame, gamma=0.8)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.equalizeHist(gray_frame)
        wajah = HaarCascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=4, minSize=(50, 50))

        for (x1, y1, width, height) in wajah:
            if width < 100 or height < 100:
                continue

            x2, y2 = x1 + width, y1 + height
            face = frame[y1:y2, x1:x2]

            try:
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                with torch.no_grad():
                    output = spoofing_model(face_tensor)
                    _, predicted = torch.max(output.data, 1)
                    is_spoof = predicted.item() == 0

                if is_spoof:
                    label = 'Spoof'
                    color = (0, 0, 255)
                    fake_count += 1
                else:
                    face_resized = face_pil.resize((160, 160))
                    face_array = asarray(face_resized)
                    face_expanded = expand_dims(face_array, axis=0)
                    signature = MyFaceNet.embeddings(face_expanded)

                    min_dist = 0.8
                    identity = 'Unknown'

                    for key, value in database.items():
                        dist = np.linalg.norm(value - signature)
                        if dist < min_dist:
                            min_dist = dist
                            identity = key
                    print(min_dist)
                    identity  =identity.split("-")[0]
                    label = identity
                    color = (0, 255, 0)
                    real_count += 1

                    if identity != 'Unknown':
                        current_time = datetime.now()
                        date_str = current_time.strftime("%d/%m/%Y")
                        time_str = current_time.strftime("%H:%M:%S")
                        df = pd.read_excel(attendance_file)

                        if not ((df['Name'] == identity) & (df['Date'] == date_str)).any():
                            new_row = {"Name": identity, "Date": date_str, "Time": time_str}
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                            df.to_excel(attendance_file, index=False)

                cv2.putText(frame, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            except Exception as e:
                print(f"Error processing face: {e}")

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')

    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed_route():
    return Response(video_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
