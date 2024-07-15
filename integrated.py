import os
import pickle
import sys
import cv2
import numpy as np
import face_recognition
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap

class FaceRecognizer:
    def __init__(self):
        self.db_file = 'face_database.pkl'
        self.load_database()
        self.gender_net = cv2.dnn.readNetFromCaffe(
            'data/deploy_gender.prototxt',
            'data/gender_net.caffemodel'
        )
        self.age_net = cv2.dnn.readNetFromCaffe(
            'data/deploy_age.prototxt',
            'data/age_net.caffemodel'
        )
        self.gender_list = ['Male', 'Female']
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def load_database(self):
        if os.path.exists(self.db_file):
            with open(self.db_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_info = data['info']
                self.face_count = data['count']
        else:
            self.known_face_encodings = []
            self.known_face_info = []
            self.face_count = 0

    def save_database(self):
        data = {
            'encodings': self.known_face_encodings,
            'info': self.known_face_info,
            'count': self.face_count
        }
        with open(self.db_file, 'wb') as f:
            pickle.dump(data, f)

    def recognize_face(self, frame):
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            gender = "Unknown"
            age = "Unknown"
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                if True in matches:
                    first_match_index = matches.index(True)
                    name, gender, age = self.known_face_info[first_match_index]
                else:
                    self.face_count += 1
                    name = f"Person_{self.face_count}"
                    face_location = face_locations[face_encodings.index(face_encoding)]
                    gender = self.estimate_gender(rgb_small_frame, face_location)
                    age = self.estimate_age(rgb_small_frame, face_location)
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_info.append((name, gender, age))
            else:
                self.face_count += 1
                name = f"Person_{self.face_count}"
                face_location = face_locations[face_encodings.index(face_encoding)]
                gender = self.estimate_gender(rgb_small_frame, face_location)
                age = self.estimate_age(rgb_small_frame, face_location)
                self.known_face_encodings.append(face_encoding)
                self.known_face_info.append((name, gender, age))
            face_names.append((name, gender, age))
        self.save_database()
        return face_locations, face_names

    def estimate_gender(self, face_image, face_location):
        top, right, bottom, left = face_location
        face = face_image[top:bottom, left:right]
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()
        gender = self.gender_list[gender_preds[0].argmax()]
        return gender

    def estimate_age(self, face_image, face_location):
        top, right, bottom, left = face_location
        face = face_image[top:bottom, left:right]
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()
        age = self.age_list[age_preds[0].argmax()]
        return age

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Face Recognition Statistics")
        self.setGeometry(100, 100, 800, 600)
        self.face_recognizer = FaceRecognizer()
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)
        self.video_label = QLabel()
        layout.addWidget(self.video_label)
        stats_layout = QVBoxLayout()
        layout.addLayout(stats_layout)
        self.total_faces_label = QLabel("Total Unique Faces: 0")
        stats_layout.addWidget(self.total_faces_label)
        self.current_face_label = QLabel("Current Face: None")
        stats_layout.addWidget(self.current_face_label)
        stats_layout.addWidget(QLabel("Recently Detected Faces:"))
        self.recent_faces_list = QListWidget()
        stats_layout.addWidget(self.recent_faces_list)
        self.cap = cv2.VideoCapture(0)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if ret:
            face_locations, face_info = self.face_recognizer.recognize_face(frame)
            for (top, right, bottom, left), (name, gender, age) in zip(face_locations, face_info):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{name}, {gender}, {age}", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)
            if face_info:
                self.current_face_label.setText(f"Current Face: {face_info[0][0]}, {face_info[0][1]}, {face_info[0][2]}")
                self.update_recent_faces(f"{face_info[0][0]}, {face_info[0][1]}, {face_info[0][2]}")
            else:
                self.current_face_label.setText("Current Face: None")
            self.total_faces_label.setText(f"Total Unique Faces: {self.face_recognizer.face_count}")
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)

    def update_recent_faces(self, face_info):
        self.recent_faces_list.insertItem(0, face_info)
        if self.recent_faces_list.count() > 10:
            self.recent_faces_list.takeItem(10)

    def closeEvent(self, event):
        self.cap.release()
        self.face_recognizer.save_database()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec_())
