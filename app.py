import os
import cv2
import face_recognition
import numpy as np
import pickle
from datetime import datetime
import pandas as pd
from flask import Flask, render_template, Response, request, redirect, url_for, flash, send_file

app = Flask(__name__)
app.secret_key = 'supersecretkey123'  # Change this in production

# Camera configuration
CAMERA_SOURCE = 0  # Default is 0 for primary camera, change if needed

# Initialize attendance system
def init_attendance_system():
    # Load trained model or create empty lists if file doesn't exist
    try:
        with open("face_encodings.pkl", "rb") as f:
            model_data = pickle.load(f)
        encodings = model_data.get("encodings", [])
        names = model_data.get("names", [])
        roll_numbers = model_data.get("roll_numbers", [])
    except (FileNotFoundError, EOFError):
        encodings = []
        names = []
        roll_numbers = []
    
    # Ensure directories exist
    os.makedirs("static/captures", exist_ok=True)
    
    # Initialize attendance files if they don't exist
    if not os.path.exists("attendance.xlsx"):
        pd.DataFrame(columns=['roll_number', 'name', 'date', 'time', 'status', 'image_path']).to_excel("attendance.xlsx", index=False)
    
    return encodings, names, roll_numbers

known_face_encodings, known_face_names, known_face_roll_numbers = init_attendance_system()
attendance_records = []

def load_attendance():
    global attendance_records
    try:
        if os.path.exists("attendance.xlsx"):
            df = pd.read_excel("attendance.xlsx")
            # Ensure proper data types
            df['roll_number'] = df['roll_number'].astype(str).str.strip()
            df['name'] = df['name'].astype(str).str.strip()
            attendance_records = df.to_dict('records')
        else:
            attendance_records = []
    except Exception as e:
        print(f"Error loading attendance: {e}")
        attendance_records = []

def save_attendance_to_excel():
    try:
        df = pd.DataFrame(attendance_records)
        # Ensure proper data types before saving
        df['roll_number'] = df['roll_number'].astype(str).str.strip()
        df['name'] = df['name'].astype(str).str.strip()
        df.to_excel("attendance.xlsx", index=False)
    except Exception as e:
        print(f"Error saving attendance: {e}")
        flash('Error saving attendance data', 'danger')

def save_attendance_to_csv():
    try:
        df = pd.DataFrame(attendance_records)
        # Ensure proper data types before saving
        df['roll_number'] = df['roll_number'].astype(str).str.strip()
        df['name'] = df['name'].astype(str).str.strip()
        df.to_csv("attendance.csv", index=False)
    except Exception as e:
        print(f"Error saving CSV: {e}")
        flash('Error saving CSV data', 'danger')

# Load existing attendance at startup
load_attendance()

def validate_roll_number(roll_number):
    """Validate and clean roll number input"""
    return str(roll_number).strip()

def validate_name(name):
    """Validate and clean name input"""
    # Remove any numbers or special characters from name
    import re
    return re.sub(r'[^a-zA-Z\s]', '', str(name).strip())

def mark_attendance(roll_number, name, frame):
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")
    
    try:
        # Clean and validate inputs
        roll_number = validate_roll_number(roll_number)
        name = validate_name(name)
        
        # Check if attendance already marked today
        existing = next((record for record in attendance_records 
                       if str(record['roll_number']).strip() == roll_number 
                       and record['date'] == date_str), None)
        
        if not existing:
            # Save captured image
            image_filename = f"{roll_number}_{date_str}_{time_str.replace(':', '-')}.jpg"
            image_path = os.path.join("static/captures", image_filename)
            
            # Convert frame to RGB before saving
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_path, rgb_frame)

            attendance_records.append({
                'roll_number': roll_number,  # Stored as cleaned string
                'name': name,                # Stored as cleaned string
                'date': date_str,
                'time': time_str,
                'status': 'Present',
                'image_path': f"captures/{image_filename}"
            })

            save_attendance_to_excel()
            save_attendance_to_csv()
            
    except Exception as e:
        print(f"Error marking attendance: {e}")
        flash('Error marking attendance', 'danger')

def generate_frames():
    camera = cv2.VideoCapture(CAMERA_SOURCE)
    if not camera.isOpened():
        print("Error: Could not open camera.")
        # Generate error frame
        error_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(error_frame, "Camera Error", (100, 240), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        ret, buffer = cv2.imencode('.jpg', error_frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        return
    
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        face_roll_numbers = []

        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.5)
            name = "Unknown"
            roll_number = "Unknown"

            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            best_match_index = np.argmin(face_distances)

            if matches[best_match_index]:
                roll_number = validate_roll_number(known_face_roll_numbers[best_match_index])
                name = validate_name(known_face_names[best_match_index])
                mark_attendance(roll_number, name, frame.copy())

            face_names.append(name)
            face_roll_numbers.append(roll_number)

        # Draw boxes and labels
        for (top, right, bottom, left), name, roll_number in zip(face_locations, face_names, face_roll_numbers):
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 255, 0), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, f"{name} ({roll_number})", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

        # Add timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/delete_record/<int:record_id>', methods=['POST'])
def delete_record(record_id):
    if 0 <= record_id < len(attendance_records):
        record = attendance_records[record_id]
        if 'image_path' in record and record['image_path']:
            try:
                image_path = os.path.join('static', record['image_path'])
                if os.path.exists(image_path):
                    os.remove(image_path)
            except Exception as e:
                print(f"Error deleting image: {e}")
        
        attendance_records.pop(record_id)
        save_attendance_to_excel()
        save_attendance_to_csv()
        flash('Record deleted successfully!', 'success')
    else:
        flash('Invalid record ID', 'danger')
    return redirect(url_for('dashboard'))

@app.route('/toggle_status/<int:record_id>', methods=['POST'])
def toggle_status(record_id):
    if 0 <= record_id < len(attendance_records):
        record = attendance_records[record_id]
        record['status'] = 'Present' if record['status'] == 'Absent' else 'Absent'
        save_attendance_to_excel()
        save_attendance_to_csv()
        flash('Status updated successfully!', 'success')
    else:
        flash('Invalid record ID', 'danger')
    return redirect(url_for('dashboard'))

@app.route('/manual_entry', methods=['POST'])
def manual_entry():
    try:
        roll_number = validate_roll_number(request.form['roll_number'])
        name = validate_name(request.form['name'])
        status = request.form['status']
        date = request.form['date']
        time = request.form['time']
        
        existing = next((record for record in attendance_records 
                       if str(record['roll_number']).strip() == roll_number 
                       and record['date'] == date), None)
        
        if existing:
            flash('Attendance for this student on this date already exists', 'warning')
        else:
            attendance_records.append({
                'roll_number': roll_number,
                'name': name,
                'date': date,
                'time': time,
                'status': status,
                'image_path': None
            })
            save_attendance_to_excel()
            save_attendance_to_csv()
            flash('Manual attendance entry added successfully!', 'success')
    except Exception as e:
        flash(f'Error adding manual entry: {str(e)}', 'danger')
    
    return redirect(url_for('dashboard'))

@app.route('/')
def home():
    return redirect(url_for('dashboard'))

@app.route('/dashboard')
def dashboard():
    sorted_records = sorted(attendance_records, 
                           key=lambda x: (x['date'], x['time']), 
                           reverse=True)
    return render_template('dashboard.html', attendance=sorted_records[:10])

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/take_attendance')
def take_attendance():
    return render_template('attendance.html')

@app.route('/download_attendance')
def download_attendance():
    save_attendance_to_excel()
    return send_file("attendance.xlsx", as_attachment=True)

@app.route('/download_csv')
def download_csv():
    save_attendance_to_csv()
    return send_file("attendance.csv", as_attachment=True)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)