import os
import cv2
import numpy as np
from deepface import DeepFace
from retinaface import RetinaFace
import subprocess

# Step 1: Ensure the output folder exists
output_folder = "output"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Step 2: Load known student names
print("Loading known student names...")
known_student_names = []
student_images_folder = "data/students"
for filename in os.listdir(student_images_folder):
    if filename.endswith(".jpg") or filename.endswith(".png"):
        # Extract student name from filename (e.g., "student1_1.jpg" -> "student1")
        student_name = filename.split("_")[0]
        if student_name not in known_student_names:
            known_student_names.append(student_name)

# Step 3: Process the video using RetinaFace for face detection and DeepFace for recognition
print("Processing video...")

# Open the video file
video_path = "data/video/classroom_light.mp4"
cap = cv2.VideoCapture(video_path)

# Get video properties for output video
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Initialize video writer for output
output_path = os.path.join(output_folder, "classroom_output.mp4")
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Process video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Detect faces using RetinaFace
    faces = RetinaFace.detect_faces(frame)
    if isinstance(faces, dict):  # Check if faces are detected
        for face_id, face_data in faces.items():
            x1, y1, x2, y2 = map(int, face_data["facial_area"])
            face = frame[y1:y2, x1:x2]

            # Ensure the face region is valid
            if face.size == 0:
                continue

            # Recognize the face using DeepFace
            try:
                recognition_result = DeepFace.find(
                    face,
                    db_path=student_images_folder,
                    enforce_detection=False,
                    silent=True  # Suppress DeepFace logs for cleaner output
                )
                if recognition_result and not recognition_result[0].empty:
                    student_name = os.path.basename(recognition_result[0].iloc[0]["identity"])
                    student_name = student_name.split("_")[0]  # Extract name from filename
                else:
                    student_name = "Unknown"

                # Draw bounding box and label
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, student_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            except Exception as e:
                print(f"Error recognizing face: {e}")

    # Write frame to output video
    out.write(frame)

    # Display the frame (optional)
    cv2.imshow("Classroom Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Video processing complete. Output saved to {output_path}")