import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime
KNOWN_FACES_DIR = "known_faces"
ATTENDANCE_FILE = "attendance.csv"
if not os.path.exists(ATTENDANCE_FILE):
    pd.DataFrame(columns=['Name', 'Time']).to_csv(ATTENDANCE_FILE, index=False)
def load_known_faces():
    known_faces = []
    known_names = []
    for filename in os.listdir(KNOWN_FACES_DIR):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_path = os.path.join(KNOWN_FACES_DIR, filename)
            image = face_recognition.load_image_file(image_path)
            encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(encoding)
            known_names.append(os.path.splitext(filename)[0])

    return known_faces, known_names
def mark_attendance(name):
    df = pd.read_csv(ATTENDANCE_FILE)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not df[(df['Name'] == name) & (df['Time'].str.contains(datetime.now().strftime("%Y-%m-%d")))].empty:
        return  # Already marked for the day
    new_entry = pd.DataFrame([[name, current_time]], columns=['Name', 'Time'])
    df = pd.concat([df, new_entry], ignore_index=True)
    df.to_csv(ATTENDANCE_FILE, index=False)
print("Loading known faces...")
known_faces, known_names = load_known_faces()
print(f"Loaded {len(known_faces)} known faces.")
cap = cv2.VideoCapture(0)
print("Starting attendance capture. Press 'q' to exit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_frame = small_frame[:, :, ::-1]
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        face_distances = face_recognition.face_distance(known_faces, face_encoding)
        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            if matches[best_match_index]:
                name = known_names[best_match_index]
                mark_attendance(name)
                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
    cv2.imshow("Attendance System", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
print("Attendance system stopped.")
