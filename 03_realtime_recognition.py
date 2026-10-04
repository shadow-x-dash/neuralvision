import cv2
import numpy as np
import os

def recognize_faces(model_path="trainer.yml"):
    if not os.path.exists(model_path):
        print(f"Error: Trained model '{model_path}' not found. Please run 02_model_training.py first.")
        return

    # Initialize recognizer and load the trained model
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read(model_path)
    
    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Names related to ids (e.g., id 1 => 'John', id 2 => 'Jane')
    # Update this dictionary mapping IDs to real names you used when creating dataset
    names = {0: 'None', 1: 'User 1', 2: 'User 2', 3: 'User 3'} 

    # Initialize webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    # Define min window size to be recognized as a face
    minW = 0.1 * cam.get(3)
    minH = 0.1 * cam.get(4)

    print("\n[INFO] Starting real-time face recognition. Press 'q' or 'ESC' to exit.")

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces in the current frame
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.2,
            minNeighbors=5,
            minSize=(int(minW), int(minH)),
        )

        for (x, y, w, h) in faces:
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Predict the id and confidence
            # LBPH predict returns the label and the distance (confidence)
            id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            
            # Check if confidence is less than 100 ==> "0" is perfect match 
            if confidence < 100:
                name = names.get(id, f"Unknown ID {id}")
                # Format confidence into a percentage representation (0 = perfect match, 100 = threshold)
                confidence_text = f"  {round(100 - confidence)}%"
            else:
                name = "Unknown"
                confidence_text = f"  {round(100 - confidence)}%"
            
            # Put text for name above the bounding box
            cv2.putText(img, str(name), (x+5, y-5), font, 1, (255, 255, 255), 2)
            
            # Put text for confidence below the bounding box
            cv2.putText(img, str(confidence_text), (x+5, y+h-5), font, 1, (255, 255, 0), 1)  
        
        cv2.imshow('Real-Time Face Recognition', img) 

        # Wait for user input - break if 'q' or ESC (key code 27) is pressed
        key = cv2.waitKey(10) & 0xff
        if key == 27 or key == ord('q'):
            break

    # Cleanup
    print("\n[INFO] Exiting Program and cleaning up...")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize_faces()
