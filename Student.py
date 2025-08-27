import cv2
import dlib
import numpy as np
import math
import time
import matplotlib.pyplot as plt
import sqlite3
from scipy.spatial import distance

#-------------Database Setup
conn = sqlite3.connect("engagement.db")  # creates file if not exists
cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS engagement_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp INTEGER,
    faces_count INTEGER,
    sleeping INTEGER,
    focused INTEGER,
    engagement_score REAL
)
""")
conn.commit()
# -------------Functions
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

def get_euler_angles(rotation_vector):
    R, _ = cv2.Rodrigues(rotation_vector)
    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0
    return math.degrees(x), math.degrees(y), math.degrees(z)  # roll, pitch, yaw

#-------------Config
PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
EYE_AR_THRESH = 0.22
SLEEP_TIME = 5  # seconds eyes must stay closed
YAW_THRESH_DEG = 20
PITCH_THRESH_DEG = 15

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)
cap = cv2.VideoCapture(0)

# Track eye-closed start times
sleep_start_times = {}

# Engagement tracking
total_frames = 0
focused_frames = 0
engagement_history = []
time_stamps = []

# 3D model points for head pose
model_points = np.array([
    (0.0, 0.0, 0.0),             # Nose tip
    (0.0, -330.0, -65.0),        # Chin
    (-225.0, 170.0, -135.0),     # Left eye left corner
    (225.0, 170.0, -135.0),      # Right eye right corner
    (-150.0, -150.0, -125.0),    # Left mouth corner
    (150.0, -150.0, -125.0)      # Right mouth corner
], dtype=np.float64)

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    h, w = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray, 0)

    #cv2.rectangle(frame, (5, 5), (180, 45), (0, 0, 0), -1)  Counter for faces, not in use currently
    #cv2.putText(frame, f"Faces: {len(faces)}", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,(0, 255, 0), 2)

    focused_in_frame = False

    for i, face in enumerate(faces):
        x1, y1, x2, y2 = face.left(), face.top(), face.right(), face.bottom()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        shape = predictor(gray, face)
        shape = np.array([[p.x, p.y] for p in shape.parts()], dtype=np.float64)

        leftEye = shape[36:42]
        rightEye = shape[42:48]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0

        if i not in sleep_start_times:
            sleep_start_times[i] = None

        sleeping = False
        if ear < EYE_AR_THRESH:
            if sleep_start_times[i] is None:
                sleep_start_times[i] = time.time()
            else:
                if time.time() - sleep_start_times[i] >= SLEEP_TIME:
                    sleeping = True
        else:
            sleep_start_times[i] = None

        status_text = "Sleeping" if sleeping else "Awake"

        # Head pose
        image_points = np.array([
            shape[30],  # Nose tip
            shape[8],   # Chin
            shape[36],  # Left eye left corner
            shape[45],  # Right eye right corner
            shape[48],  # Left Mouth corner
            shape[54]   # Right mouth corner
        ], dtype=np.float64)

        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype="double"
        )
        dist_coeffs = np.zeros((4, 1))

        focused_text = "Unknown"
        try:
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
            if success:
                roll_deg, pitch_deg, yaw_deg = get_euler_angles(rotation_vector)
                if (abs(yaw_deg) <= YAW_THRESH_DEG) and (abs(pitch_deg) <= PITCH_THRESH_DEG):
                    focused_text = "Focused"
                    if not sleeping:
                        focused_in_frame = True
                else:
                    focused_text = "Not Focused"
            else:
                focused_text = "PoseFail"
        except:
            focused_text = "PoseErr"

        display_label = "Sleeping" if sleeping else f"{focused_text} | {status_text}"
        cv2.rectangle(frame, (x1, y1 - 22), (x2, y1), (0, 0, 0), -1)
        cv2.putText(frame, display_label, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,(255, 255, 255), 1)

    # Engagement stats
    total_frames += 1
    if focused_in_frame:
        focused_frames += 1
    engagement_score = (focused_frames / total_frames) * 100 if total_frames > 0 else 0

    # Save history every second + log to DB
    if int(time.time() - start_time) > len(time_stamps):
        ts = int(time.time() - start_time)
        time_stamps.append(ts)
        engagement_history.append(engagement_score)

        # Insert into database
        cursor.execute("""
                       INSERT INTO engagement_log (timestamp, faces_count, sleeping, focused, engagement_score)
                       VALUES (?, ?, ?, ?, ?)
                       """, (
                           ts,
                           len(faces),
                           int(any([sleep_start_times[i] is not None for i in sleep_start_times])),  # 1 if any sleeping
                           int(focused_in_frame),
                           engagement_score
                       ))
        conn.commit()

    # Draw on screen
    cv2.rectangle(frame, (5, 5), (260, 45), (0, 0, 0), -1)
    cv2.putText(frame, f"Engagement: {engagement_score:.1f}%", (12, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Face Monitor", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
conn.close()

#-------------Engagement Graph
plt.figure(figsize=(8, 4))
plt.plot(time_stamps, engagement_history, marker="o")
plt.title("Engagement Over Time")
plt.xlabel("Time (seconds)")
plt.ylabel("Engagement Score (%)")
plt.ylim(0, 100)
plt.grid(True)
plt.show()
