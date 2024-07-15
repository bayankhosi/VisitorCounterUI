#Step 1
import inception_resnet_v1

#Step 2
model = inception_resnet_v1.InceptionResNetV1()
model.load_weights('keras-facenet-h5/facenet_keras_weights.h5')

# Fix from https://community.deeplearning.ai/t/problems-when-i-implemented-my-own-face-recognition-projects/396496/13 when facenet_keras not loading

import cv2
from mtcnn import MTCNN
from keras.models import load_model
import numpy as np
import sqlite3
from PyQt5 import QtWidgets, QtCore
import sys
import threading

# Initialize face detector and models
detector = MTCNN()
facenet_model = load_model('facenet_keras.h5')
age_model = load_model('age_model.h5')
gender_model = load_model('gender_model.h5')

# Database setup
conn = sqlite3.connect('faces.db')
c = conn.cursor()
c.execute('''CREATE TABLE IF NOT EXISTS faces
             (id INTEGER PRIMARY KEY, embedding BLOB, age INTEGER, gender TEXT, timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

def detect_faces(frame):
    return detector.detect_faces(frame)

def extract_embeddings(face):
    face = cv2.resize(face, (160, 160))
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    samples = np.expand_dims(face, axis=0)
    embeddings = facenet_model.predict(samples)
    return embeddings[0]

def estimate_age(face):
    face = cv2.resize(face, (64, 64))
    face = face.astype('float32')
    face = face / 255.0
    samples = np.expand_dims(face, axis=0)
    age = age_model.predict(samples)
    return int(age[0][0])

def estimate_gender(face):
    face = cv2.resize(face, (64, 64))
    face = face.astype('float32')
    face = face / 255.0
    samples = np.expand_dims(face, axis=0)
    gender = gender_model.predict(samples)
    return 'Male' if gender[0][0] > 0.5 else 'Female'

def store_face(embedding, age, gender):
    c.execute("INSERT INTO faces (embedding, age, gender) VALUES (?, ?, ?)", (embedding.tobytes(), age, gender))
    conn.commit()

def get_unique_face_count():
    c.execute("SELECT COUNT(DISTINCT embedding) FROM faces")
    return c.fetchone()[0]

class FaceStatsApp(QtWidgets.QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        
    def initUI(self):
        self.setGeometry(100, 100, 400, 300)
        self.setWindowTitle('Face Stats')
        
        self.label = QtWidgets.QLabel(self)
        self.label.setText("Unique Faces: 0")
        self.label.move(20, 20)
        
        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self.update_stats)
        self.timer.start(1000)  # Update every second

        self.show()

    def update_stats(self):
        unique_faces = get_unique_face_count()
        self.label.setText(f"Unique Faces: {unique_faces}")

def process_video():
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        faces = detect_faces(frame)
        for face in faces:
            x, y, width, height = face['box']
            face_img = frame[y:y+height, x:x+width]
            embeddings = extract_embeddings(face_img)
            age = estimate_age(face_img)
            gender = estimate_gender(face_img)
            store_face(embeddings, age, gender)
            cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)

        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    ex = FaceStatsApp()

    # Start video processing in a separate thread
    video_thread = threading.Thread(target=process_video)
    video_thread.start()

    sys.exit(app.exec_())
