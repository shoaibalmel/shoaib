from flask import Flask, render_template, jsonify, Response
import cv2
import numpy as np
import threading
import time
from datetime import datetime

app = Flask(__name__)

# Define parking slots [(x, y, width, height)]
parking_slots = [
    (50, 100, 100, 150), (200, 100, 100, 150), (350, 100, 100, 150),  # First row
    (50, 300, 100, 150), (200, 300, 100, 150), (350, 300, 100, 150)   # Second row
]
slot_status = [False] * 6  # Track occupied slots
entry_times = [None] * 6    # Store timestamps when occupied
charges_per_hour = 50       # Define charge rate per hour

def detect_parking_slots(frame):
    """Detect empty/occupied slots and update status."""
    global slot_status, entry_times

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    new_status = []
    for i, (x, y, w, h) in enumerate(parking_slots):
        roi = edges[y:y+h, x:x+w]
        white_pixels = cv2.countNonZero(roi)

        occupied = white_pixels > 800  # Adjust threshold for accuracy
        new_status.append(occupied)

        # Track entry time
        if occupied and not slot_status[i]:  
            entry_times[i] = datetime.now()  # Record when car enters
        elif not occupied and slot_status[i]:  
            if entry_times[i]:  
                exit_time = datetime.now()
                parked_duration = (exit_time - entry_times[i]).total_seconds() / 3600
                charge = round(parked_duration * charges_per_hour, 2)
                print(f"Slot {i+1} - Duration: {round(parked_duration, 2)} hours, Charge: ₹{charge}")
                entry_times[i] = None  # Reset entry time

        # Draw rectangle
        color = (0, 0, 255) if occupied else (0, 255, 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 3)
        cv2.putText(frame, f"Slot {i+1}", (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    slot_status = new_status  # Update global slot status
    return frame

def generate_camera_feed():
    """Capture frames and stream to the web page."""
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = detect_parking_slots(frame)
        _, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()

@app.route('/status')
def status():
    """Return the parking status with charges."""
    status_info = []
    for i in range(len(parking_slots)):
        duration = 0
        charge = 0
        if slot_status[i] and entry_times[i]:
            duration = (datetime.now() - entry_times[i]).total_seconds() / 3600
            charge = round(duration * charges_per_hour, 2)

        status_info.append({
            "slot": i+1,
            "occupied": slot_status[i],
            "parked_time_hours": round(duration, 2),
            "charge": charge
        })

    return jsonify({
        "total_slots": len(parking_slots),
        "occupied_slots": sum(slot_status),
        "available_slots": len(parking_slots) - sum(slot_status),
        "slot_details": status_info
    })

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_camera_feed(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, port=5000)
