import cv2
import torch
import numpy as np
from PIL import Image
from numpy import asarray, expand_dims
from facenet_pytorch import InceptionResnetV1
from torchvision import transforms
from model import initialize_model  
import matplotlib.pyplot as plt
from keras_facenet import FaceNet
from datetime import datetime
import pandas as pd
import pickle
import os

# Load Haar cascade for face detection
HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Initialize face recognition model (InceptionResnetV1)
MyFaceNet = FaceNet()

# Initialize the anti-spoofing model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
spoofing_model = initialize_model(num_classes=2).to(device)
spoofing_model.load_state_dict(torch.load('anti_spoofing_model1.pth', map_location=device))
spoofing_model.eval()

# Define image transformations for the anti-spoofing model
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Example face embeddings database (replace with actual embeddings)
myfile = open("data1.pkl", "rb")
database = pickle.load(myfile)
myfile.close()

# OpenCV video capture
cap = cv2.VideoCapture(0)


# Create an attendance Excel file if it doesn't exist
attendance_file = "attendance.xlsx"
if not os.path.exists(attendance_file):
    df = pd.DataFrame(columns=["Name", "Date", "Time"])
    df.to_excel(attendance_file, index=False)

# Initialize counters for real and fake detections
real_count = 0
fake_count = 0

# Setup matplotlib for live bar graph and accuracy metrics
#plt.ion()  # Turn interactive mode on for live updating plots
#fig, ax = plt.subplots()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame. Exiting...")
        break

    wajah = HaarCascade.detectMultiScale(frame, 1.1, 4)

    if len(wajah) > 0:
        for i, (x1, y1, width, height) in enumerate(wajah):
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            # Extract and preprocess the face for the anti-spoofing model
            face = frame[y1:y2, x1:x2]
            if face.size == 0:
                continue

            try:
                face_pil = Image.fromarray(cv2.cvtColor(face, cv2.COLOR_BGR2RGB))
                face_tensor = transform(face_pil).unsqueeze(0).to(device)

                # Predict spoofing
                with torch.no_grad():
                    output = spoofing_model(face_tensor)
                    _, predicted = torch.max(output.data, 1)
                    is_spoof = predicted.item() == 0  # Assuming 0 is for spoof and 1 for real

                if is_spoof:
                    label = 'Spoof'
                    color = (0, 0, 255)
                    fake_count += 1  # Increment fake face count
                else:
                    # Perform face recognition only if the face is real
                    face_resized = face_pil.resize((160, 160))
                    face_array = asarray(face_resized)
                    face_expanded = expand_dims(face_array, axis=0)

                    # Get embeddings using the face recognition model
                    signature = MyFaceNet.embeddings(face_expanded)

                    # Initialize minimum distance and identity
                    min_dist = 0.83
                    identity = 'Unknown'

                    # Compare with the database
                    for key, value in database.items():
                        dist = np.linalg.norm(value - signature)
                        if dist < min_dist:
                            min_dist = dist
                            identity = key
                    identity=identity.split("-")[0]
                    label = identity
                    color = (0, 255, 0)
                    real_count += 1  # Increment real face count

                    print(min_dist)
                    # Log attendance if the person is recognized
                    if identity != 'Unknown':
                        current_time = datetime.now()
                        date_str = current_time.strftime("%d/%m/%Y")
                        time_str = current_time.strftime("%H:%M:%S")

                        # Load the existing attendance file
                        df = pd.read_excel(attendance_file)

                        # Check if attendance for this person on this date is already logged
                        if not ((df['Name'] == identity) & (df['Date'] == date_str)).any():
                            # Add a new row for attendance
                            new_row = {"Name": identity, "Date": date_str, "Time": time_str}
                            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)

                            # Save the updated DataFrame back to the Excel file
                            df.to_excel(attendance_file, index=False)
                            
                cv2.putText(frame, label, (x1 + 10, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            except Exception as e:
                print(f"Error processing face: {e}")

    
    
    # Display the frame
    cv2.imshow('Face Anti-Spoofing and Recognition', frame)

    # Exit on 'ESC' key press
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
