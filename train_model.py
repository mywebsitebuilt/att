import os
import face_recognition
import pickle
from collections import defaultdict

def train_model():
    known_encodings = []
    known_names = []
    known_roll_numbers = []
    
    # Path to the directory containing student folders
    base_dir = "static/known_faces"
    
    # Walk through each student's folder
    for root, dirs, files in os.walk(base_dir):
        # Get roll number from folder name
        roll_number = os.path.basename(root)
        
        if roll_number == "known_faces":
            continue  # Skip the root directory
            
        # Process each image in the student's folder
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(root, file)
                
                # Load image
                image = face_recognition.load_image_file(image_path)
                
                # Get face encodings
                face_encodings = face_recognition.face_encodings(image)
                
                if len(face_encodings) > 0:
                    # Add encoding for each face found in the image
                    for encoding in face_encodings:
                        known_encodings.append(encoding)
                        known_names.append(roll_number)  # Using roll number as identifier
                        known_roll_numbers.append(roll_number)
    
    # Save the trained model
    model_data = {
        "encodings": known_encodings,
        "names": known_names,
        "roll_numbers": known_roll_numbers
    }
    
    with open("face_encodings.pkl", "wb") as f:
        pickle.dump(model_data, f)
    
    print(f"Training complete. {len(known_names)} face encodings saved.")

if __name__ == "__main__":
    train_model()