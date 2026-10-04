import cv2
import os

def create_dataset(user_id, num_samples=10, save_dir="dataset"):
    # Create directory if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    # Initialize webcam
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("Error: Could not open webcam. Ensure it's connected and accessible.")
        return

    # Load Haar cascade for face detection
    # Using OpenCV's built-in Haar cascades
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    print(f"\n[INFO] Initializing face capture. Look at the camera and wait...")
    count = 0

    while True:
        ret, img = cam.read()
        if not ret:
            print("Error: Failed to capture image from webcam.")
            break
            
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.3, 
            minNeighbors=5
        )

        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
            count += 1
            
            # Save the captured face into the dataset folder
            # Naming convention: User.<ID>.<Count>.jpg
            file_name = f"{save_dir}/User.{user_id}.{count}.jpg"
            cv2.imwrite(file_name, gray[y:y+h, x:x+w])
            
            # Display the video frame with bounding box
            cv2.imshow('Dataset Creation - Face Capture', img)

        # Press 'q' to stop or wait until we have the required number of samples
        key = cv2.waitKey(100) & 0xff 
        if key == ord('q'):
            break
        elif count >= num_samples:
             break

    print(f"\n[INFO] Exiting Program and cleanup. {count} samples collected.")
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    user_input = input("Enter a numeric user ID (e.g. 1): ")
    try:
        user_id = int(user_input)
        create_dataset(user_id)
    except ValueError:
        print("Error: User ID must be an integer number.")
