import cv2
import os
import json
import threading
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, Response, request, jsonify

app = Flask(__name__)

# Initialize model and cascade
model_path = "trainer.yml"
recognizer = cv2.face.LBPHFaceRecognizer_create()
model_loaded = False

def load_model():
    global model_loaded
    if os.path.exists(model_path):
        recognizer.read(model_path)
        model_loaded = True
    else:
        model_loaded = False

load_model()

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

# Load users
USERS_FILE = 'users.json'
if not os.path.exists(USERS_FILE):
    with open(USERS_FILE, 'w') as f:
        # Pre-populate with dummy values if needed, or empty
        json.dump({"1": "User 1", "2": "User 2", "3": "User 3"}, f)

def get_users():
    with open(USERS_FILE, 'r') as f:
        return json.load(f)

# Global variables for capturing
capture_mode = False
capture_id = None
capture_count = 0
capture_total = 10
dataset_dir = "dataset"

if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# Attendance
attendance_file = "attendance.csv"
if not os.path.exists(attendance_file):
    with open(attendance_file, "w") as f:
        f.write("Name,Time\n")

marked_attendance = set()

import base64
import numpy as np

def process_frame_logic(img_data):
    global capture_mode, capture_id, capture_count, model_loaded
    
    # Decode base64 to OpenCV image
    if ',' in img_data:
        img_data = img_data.split(',')[1]
    
    img_bytes = base64.b64decode(img_data)
    np_arr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    
    if img is None:
        return ""

    minW = 0.1 * img.shape[1]
    minH = 0.1 * img.shape[0]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH)),
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 75, 100), 2)
        
        if capture_mode:
            import time
            global last_capture_time, capture_paused
            if 'last_capture_time' not in globals():
                last_capture_time = 0
            if 'capture_paused' not in globals():
                capture_paused = False
                
            percentage = capture_count / capture_total if capture_total > 0 else 0
            if percentage < 0.2:
                instr = "Look STRAIGHT"
            elif percentage < 0.4:
                instr = "Turn LEFT"
            elif percentage < 0.6:
                instr = "Turn RIGHT"
            elif percentage < 0.8:
                instr = "Tilt UP"
            else:
                instr = "Tilt DOWN"

            if not capture_paused:
                current_time = time.time()
                if current_time - last_capture_time > 0.15:
                    capture_count += 1
                    last_capture_time = current_time
                    file_name = f"{dataset_dir}/User.{capture_id}.{capture_count}.jpg"
                    cv2.imwrite(file_name, gray[y:y+h, x:x+w])
                    
                    if capture_count % 10 == 0 and capture_count < capture_total:
                        capture_paused = True
            else:
                cv2.putText(img, "PAUSED - Click Next", (x+5, y+h+55), font, 0.7, (0, 0, 255), 2)
            
            cv2.putText(img, f"Capturing: {capture_count}/{capture_total}", (x+5, y-10), font, 0.8, (0, 255, 255), 2)
            cv2.putText(img, instr, (x+5, y+h+25), font, 0.9, (0, 255, 255), 2)
            
            if capture_count >= capture_total:
                capture_mode = False
                def train_and_reload():
                    from importlib import import_module
                    train_module = import_module('02_model_training')
                    train_module.train_model()
                    load_model()
                threading.Thread(target=train_and_reload).start()
        else:
            if model_loaded:
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                users = get_users()
                
                accuracy = round(100 - confidence)
                if accuracy >= 45:
                    name = users.get(str(id), f"Unknown ID {id}")
                    confidence_text = f"Accuracy: {accuracy}%"
                    status_text = ""
                    
                    if name != "Unknown" and "Unknown" not in name:
                        today_str = datetime.now().strftime("%Y-%m-%d")
                        now_str = datetime.now().strftime("%H:%M:%S")
                        attendance_key = f"{name}_{today_str}"
                        
                        if attendance_key not in marked_attendance:
                            with open(attendance_file, "a") as f:
                                f.write(f"{name},{today_str} {now_str}\n")
                            marked_attendance.add(attendance_key)
                        
                        status_text = "Attendance Done"
                else:
                    name = "Unregistered Face"
                    acc_display = max(0, accuracy)
                    confidence_text = f"Accuracy: {acc_display}%"
                    status_text = ""
                
                cv2.putText(img, str(name), (x+5, y-10), font, 0.8, (0, 0, 0), 3)
                cv2.putText(img, str(name), (x+5, y-10), font, 0.8, (255, 255, 255), 2)
                
                if status_text:
                    cv2.putText(img, status_text, (x+5, y-35), font, 0.8, (0, 0, 0), 3)
                    cv2.putText(img, status_text, (x+5, y-35), font, 0.8, (0, 255, 0), 2)
                
                cv2.putText(img, str(confidence_text), (x+5, y+h+20), font, 0.6, (0, 0, 0), 3)
                cv2.putText(img, str(confidence_text), (x+5, y+h+20), font, 0.6, (255, 255, 0), 1)
            else:
                cv2.putText(img, "Model Not Trained", (x+5, y-10), font, 0.8, (0, 0, 255), 2)

    ret, buffer = cv2.imencode('.jpg', img)
    if not ret:
        return ""
    encoded_img = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{encoded_img}"

@app.route('/')
def index():
    return render_template('index.html', model_loaded=model_loaded)

@app.route('/process_frame', methods=['POST'])
def process_frame():
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "No image data"}), 400
        
    result_image = process_frame_logic(data['image'])
    return jsonify({"image": result_image})

@app.route('/add_user', methods=['POST'])
def add_user():
    global capture_mode, capture_id, capture_count, capture_total
    
    if capture_mode:
        return jsonify({"status": "error", "message": "Already capturing data for a user."})
        
    data = request.json
    name = data.get('name')
    frames = data.get('frames', 10)
    
    try:
        frames = int(frames)
        if frames < 1:
            frames = 10
    except ValueError:
        frames = 10
        
    if not name:
        return jsonify({"status": "error", "message": "Name is required."})
        
    users = get_users()
    new_id = 1
    if users:
        try:
            new_id = max([int(k) for k in users.keys()]) + 1
        except ValueError:
            new_id = 1
            
    users[str(new_id)] = name
    with open(USERS_FILE, 'w') as f:
        json.dump(users, f)
        
    capture_id = new_id
    capture_count = 0
    capture_total = frames
    capture_mode = True
    
    return jsonify({"status": "success", "message": f"Started capturing data for {name}. Please look at the camera."})

@app.route('/attendance_data')
def attendance_data():
    if not os.path.exists(attendance_file):
        return jsonify([])
    
    try:
        df = pd.read_csv(attendance_file)
        # Reverse to show latest first
        df = df.iloc[::-1]
        return jsonify(df.to_dict(orient='records'))
    except Exception as e:
        return jsonify([])

@app.route('/get_users_list')
def get_users_list():
    users = get_users()
    return jsonify([{"id": k, "name": v} for k, v in users.items()])

@app.route('/edit_user', methods=['POST'])
def edit_user():
    data = request.json
    user_id = str(data.get('id'))
    new_name = data.get('name')
    
    if not user_id or not new_name:
        return jsonify({"status": "error", "message": "ID and name are required."})
        
    users = get_users()
    if user_id in users:
        users[user_id] = new_name
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
        return jsonify({"status": "success", "message": "User updated successfully."})
    else:
        return jsonify({"status": "error", "message": "User not found."})

@app.route('/delete_user', methods=['POST'])
def delete_user():
    data = request.json
    user_id = str(data.get('id'))
    
    if not user_id:
        return jsonify({"status": "error", "message": "ID is required."})
        
    users = get_users()
    if user_id in users:
        del users[user_id]
        with open(USERS_FILE, 'w') as f:
            json.dump(users, f)
            
        # Delete images for this user
        import glob
        import os
        import threading
        
        for file in glob.glob(f"{dataset_dir}/User.{user_id}.*.jpg"):
            try:
                os.remove(file)
            except Exception:
                pass
                
        # Trigger training to remove the user from the model
        def train_and_reload():
            from importlib import import_module
            train_module = import_module('02_model_training')
            train_module.train_model()
            load_model()
        
        threading.Thread(target=train_and_reload).start()

        return jsonify({"status": "success", "message": "User and data deleted successfully."})
    else:
        return jsonify({"status": "error", "message": "User not found."})

@app.route('/capture_status')
def capture_status():
    global capture_paused
    if 'capture_paused' not in globals():
        capture_paused = False
    return jsonify({
        "capture_mode": capture_mode,
        "capture_count": capture_count,
        "capture_total": capture_total,
        "capture_paused": capture_paused
    })

@app.route('/resume_capture', methods=['POST'])
def resume_capture():
    global capture_paused
    capture_paused = False
    return jsonify({"status": "success"})

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
