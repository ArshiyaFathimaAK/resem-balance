import cv2
import mediapipe as mp
import numpy as np
import os
from deepface import DeepFace

# --- Initialize MediaPipe Face Mesh ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1)
mp_draw = mp.solutions.drawing_utils
draw_spec = mp_draw.DrawingSpec(thickness=1, circle_radius=1)

# --- Load reference faces embeddings ---
reference_dir = "reference_faces"  # folder containing character images
reference_embeddings = {}
for file in os.listdir(reference_dir):
    path = os.path.join(reference_dir, file)
    name = os.path.splitext(file)[0]
    try:
        emb = DeepFace.represent(path, model_name="Facenet")[0]["embedding"]
        reference_embeddings[name] = np.array(emb)
    except:
        print(f"Face not detected in {file}")

# --- Start webcam ---
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(img_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Draw landmarks
            mp_draw.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION,
                                   draw_spec, draw_spec)
            
            # Crop the face for DeepFace
            h, w, _ = frame.shape
            xs = [int(lm.x * w) for lm in face_landmarks.landmark]
            ys = [int(lm.y * h) for lm in face_landmarks.landmark]
            x_min, x_max = max(min(xs) - 20, 0), min(max(xs) + 20, w)
            y_min, y_max = max(min(ys) - 20, 0), min(max(ys) + 20, h)
            face_crop = frame[y_min:y_max, x_min:x_max]
            
            try:
                user_emb = DeepFace.represent(face_crop, model_name="Facenet")[0]["embedding"]
                # Compute cosine similarity
                similarities = {}
                for name, ref_emb in reference_embeddings.items():
                    sim = np.dot(user_emb, ref_emb) / (np.linalg.norm(user_emb) * np.linalg.norm(ref_emb))
                    similarities[name] = sim * 100  # convert to percentage
                
                # Sort and display top 3
                top_matches = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:3]
                y0 = 30
                for name, score in top_matches:
                    text = f"{name}: {score:.2f}%"
                    cv2.putText(frame, text, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 255, 0), 2)
                    y0 += 30
            except:
                pass

    cv2.imshow("Face Similarity", frame)
    if cv2.waitKey(1) & 0xFF == 27:  # press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
