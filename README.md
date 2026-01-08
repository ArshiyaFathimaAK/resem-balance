# 🎭 Resem-Balance

A real-time **face similarity tracker** that analyzes facial features and compares them against user-provided reference images to estimate resemblance percentages.
Built using **OpenCV**, **MediaPipe**, and **Deep Learning–based face embeddings**.

---

## 🚀 Features

* 📸 **Two input modes**

  * Upload an image from your device
  * Real-time webcam face detection

* 🧠 **Facial feature extraction** using MediaPipe Face Mesh

* 📊 **Similarity scoring** against user-provided reference images

* 🟦 **Live face bounding box & tracking**

* 🔢 Displays resemblance percentages (even if similarity is low)

---

## 🛠 Tech Stack

* **Python 3.11**
* **OpenCV** – video capture & face bounding box
* **MediaPipe** – facial landmark detection
* **DeepFace / Deep Learning models** – feature embeddings
* **NumPy** – vector operations

---

## 📂 Project Structure

```
resem-balance/
│
├── face-env/                 # Optional: virtual environment
├── reference_faces/          # Folder where users add their reference images
├── face_similarity.py        # Main application script
└── README.md
```

> Users should add their own reference images inside the `reference_faces` folder before running the app.

---

## ⚙️ Installation

1. **Clone the repository**

```bash
git clone https://github.com/ArshiyaFathimaAK/resem-balance.git
cd resem-balance
```

2. **Install dependencies**

```bash
pip install mediapipe==0.10.21 protobuf==4.25.3 opencv-python deepface torch torchvision mtcnn tf-keras
```

> ⚠️ Note: Dependency versions are pinned to avoid protobuf and TensorFlow conflicts.

---

## ▶️ Usage

1. Add your reference images inside the `reference_faces` folder.
2. Run the application:

```bash
python face_similarity.py
```

3. Allow camera access when prompted.
4. The app will detect your face and display similarity percentages in real time.

---

## 📌 Example Output

```
Reference1: 24.3%
Reference2: 21.7%
Reference3: 18.9%
...
```

---

## ⚠️ Disclaimer

This project is for **educational and entertainment purposes only**.
The similarity percentages are **approximate** and **do not represent official biometric identification**.
Use responsibly and respect privacy when comparing faces.

---

## 🌱 Future Improvements

* UI with buttons for **Upload Image / Webcam Mode**
* Better normalization for similarity scores
* Model fine-tuning for higher accuracy
* Web version using **Flask** or **Streamlit**

---

## 👩‍💻 Author

**Arshiya Fathima A K**
Computer Science Engineering
LinkedIn: [https://www.linkedin.com/in/arshiya-fathima-ak](https://www.linkedin.com/in/arshiya-fathima-ak)

---

