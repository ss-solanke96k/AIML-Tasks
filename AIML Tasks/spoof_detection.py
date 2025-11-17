import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
print("Motion-based spoof detection started. Press 'q' to quit.")

resize_dim = (96, 96)
motion_threshold = 7
stable_frames_required = 20
min_face_size = 60
min_brightness = 50

prev_faces = {}
motion_history = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if np.mean(gray) < min_brightness:
        cv2.putText(frame, "Lighting too low — please brighten your environment",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Face Spoof Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
        continue

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for i, (x, y, w, h) in enumerate(faces):

        if w < min_face_size or h < min_face_size:
            continue

        face_id = f"face_{i}"
        face_region = gray[y:y + h, x:x + w]
        face_resized = cv2.resize(face_region, resize_dim)

        if face_id not in motion_history:
            motion_history[face_id] = {"scores": [], "label": "Spoof"}

        if face_id in prev_faces:
            prev_resized = prev_faces[face_id]
            diff = cv2.absdiff(prev_resized, face_resized)
            motion_score = np.sum(diff) / (resize_dim[0] * resize_dim[1])

            motion_history[face_id]["scores"].append(motion_score)
            if len(motion_history[face_id]["scores"]) > stable_frames_required:
                motion_history[face_id]["scores"].pop(0)

            avg_motion = np.mean(motion_history[face_id]["scores"])

            if avg_motion > motion_threshold:
                motion_history[face_id]["label"] = "Real"
            else:
                motion_history[face_id]["label"] = "Spoof"

        prev_faces[face_id] = face_resized.copy()

        label = motion_history[face_id]["label"]
        color = (0, 255, 0) if label == "Real" else (0, 0, 255)

        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        cv2.putText(frame, f"{label}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        if face_id in motion_history and motion_history[face_id]["scores"]:
            avg_motion = np.mean(motion_history[face_id]["scores"])
            cv2.putText(frame, f"Motion: {avg_motion:.2f}", (x, y + h + 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.putText(frame, "Please move your head slightly", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("Face Spoof Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()