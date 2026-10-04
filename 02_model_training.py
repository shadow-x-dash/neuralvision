import cv2
import numpy as np
import os
from PIL import Image

def train_model(dataset_dir="dataset", model_save_path="trainer.yml"):
    if not os.path.exists(dataset_dir):
        print(f"Error: Dataset directory '{dataset_dir}' not found. Please run 01_dataset_creation.py first.")
        return

    # Initialize LBPH face recognizer
    # LBPH (Local Binary Patterns Histograms) is used because it is robust against 
    # illumination variations and computationally efficient for real-time.
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    
    # Load Haar cascade for detection (used here again to ensure bounding box alignment during training)
    detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    def get_images_and_labels(path):
        image_paths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jpg')]
        face_samples = []
        ids = []

        for image_path in image_paths:
            # Convert image to grayscale using PIL
            PIL_img = Image.open(image_path).convert('L')
            img_numpy = np.array(PIL_img, 'uint8')

            # Extract the user ID from the filename (e.g., User.1.25.jpg -> ID 1)
            try:
                id = int(os.path.split(image_path)[-1].split(".")[1])
            except (IndexError, ValueError):
                continue
                
            # IMPORTANT FIX: The images saved in the dataset by app.py are ALREADY cropped faces.
            # Running Haar Cascade again here often fails to find a face inside a tightly cropped face,
            # causing the system to throw away 90% of your training images!
            # We now use 100% of the images directly.
            face_samples.append(img_numpy)
            ids.append(id)

        return face_samples, ids

    print("\n[INFO] Training faces. It will take a few seconds. Wait...")
    faces, ids = get_images_and_labels(dataset_dir)
    
    if len(faces) == 0:
        print("Error: No training data found.")
        return

    # Train the model with the collected faces and labels
    recognizer.train(faces, np.array(ids))

    # Save the trained model to a .yml file
    recognizer.write(model_save_path)
    
    # Print the number of unique faces trained
    unique_ids = len(np.unique(ids))
    print(f"\n[INFO] {unique_ids} faces trained. Model saved as '{model_save_path}'.")

if __name__ == "__main__":
    train_model()
