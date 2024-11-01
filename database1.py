import os
from PIL import Image as Img
from numpy import asarray, expand_dims
from keras_facenet import FaceNet
import pickle
import cv2

# Initialize FaceNet and Haar cascade
HaarCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
MyFaceNet = FaceNet()

# Root folder containing subfolders with images
root_folder = 'photos1/'
database = {}

# Loop through each subfolder in the root folder
for subfolder in os.listdir(root_folder):
    subfolder_path = os.path.join(root_folder, subfolder)
    
    # Check if it is a directory
    if os.path.isdir(subfolder_path):
        # Counter for each image in the subfolder
        image_count = 1
        
        # Loop through each image in the subfolder
        for filename in os.listdir(subfolder_path):
            image_path = os.path.join(subfolder_path, filename)
            print(f"Processing {filename} in folder {subfolder}")
            
            # Read the image
            img = cv2.imread(image_path)
            
            # Detect face
            wajah = HaarCascade.detectMultiScale(img, 1.1, 4)
            
            # Get coordinates of the detected face, or default values if no face found
            if len(wajah) > 0:
                x1, y1, width, height = wajah[0]         
            else:
                x1, y1, width, height = 1, 1, 10, 10
                
            x1, y1 = abs(x1), abs(y1)
            x2, y2 = x1 + width, y1 + height

            if img is None:
                continue
            
            # Convert image to RGB and PIL format for processing
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_pil = Img.fromarray(img_rgb)
            img_array = asarray(img_pil)
            
            # Crop the face from the image
            face = img_array[y1:y2, x1:x2]
            
            # Resize and preprocess the face for embedding
            face_pil = Img.fromarray(face).resize((160, 160))
            face_array = asarray(face_pil)
            face_array = expand_dims(face_array, axis=0)
            
            # Get face embedding
            signature = MyFaceNet.embeddings(face_array)
            
            # Create unique key for this face using the subfolder name and image count
            key = f"{subfolder}-{image_count}"
            database[key] = signature
            
            # Increment image counter for the subfolder
            image_count += 1

# Save the database dictionary to a file
with open("data1.pkl", "wb") as myfile:
    pickle.dump(database, myfile)

print("Image embeddings saved successfully!")
