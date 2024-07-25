#pip install opencv-python-headless face_recognition pyqt5
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
        # Initialize face database file and load existing data if available
        self.db_file = 'face_database.pkl'
        self.load_database()

        # Load pre-trained models for gender and age detection
        self.gender_net = cv2.dnn.readNetFromCaffe(
            'data/deploy_gender.prototxt',
            'data/gender_net.caffemodel'
        )
        self.age_net = cv2.dnn.readNetFromCaffe(
            'data/deploy_age.prototxt',
            'data/age_net.caffemodel'
        )

        # Lists for gender and age categories
        self.gender_list = ['Male', 'Female']
        self.age_list = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']

    def load_database(self):
        # Load existing face database from pickle file if it exists
        if os.path.exists(self.db_file):
            with open(self.db_file, 'rb') as f:
                data = pickle.load(f)
                self.known_face_encodings = data['encodings']
                self.known_face_info = data['info']
                self.face_count = data['count']
        else:
            # Initialize empty database if no file exists
            self.known_face_encodings = []
            self.known_face_info = []
            self.face_count = 0

    def save_database(self):
        # Save current face database to pickle file
        data = {
            'encodings': self.known_face_encodings,
            'info': self.known_face_info,
            'count': self.face_count
        }
        with open(self.db_file, 'wb') as f:
            pickle.dump(data, f)

    def recognize_face(self, frame):
        # Resize frame for faster processing
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Locate faces and encode them
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        # List to store recognized face names, genders, and ages
        face_names = []
        for face_encoding in face_encodings:
            name = "Unknown"
            gender = "Unknown"
            age = "Unknown"

            # Check if any known faces match the current face
            if self.known_face_encodings:
                matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                if True in matches:
                    first_match_index = matches.index(True)
                    name, gender, age = self.known_face_info[first_match_index]
                else:
                    # Increment face count and assign default name for new faces
                    self.face_count += 1
                    name = f"Person_{self.face_count}"

                    # Estimate gender and age for new face
                    face_location = face_locations[face_encodings.index(face_encoding)]
                    gender = self.estimate_gender(rgb_small_frame, face_location)
                    age = self.estimate_age(rgb_small_frame, face_location)

                    # Add new face to database
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_info.append((name, gender, age))
            else:
                # Initialize database with first detected face
                self.face_count += 1
                name = f"Person_{self.face_count}"

                # Estimate gender and age for first face
                face_location = face_locations[face_encodings.index(face_encoding)]
                gender = self.estimate_gender(rgb_small_frame, face_location)
                age = self.estimate_age(rgb_small_frame, face_location)

                # Add first face to database
                self.known_face_encodings.append(face_encoding)
                self.known_face_info.append((name, gender, age))

            # Append recognized face information to list
            face_names.append((name, gender, age))

        # Save database after recognizing all faces in the current frame
        self.save_database()

        return face_locations, face_names

    def estimate_gender(self, face_image, face_location):
        # Extract face region from image
        top, right, bottom, left = face_location
        face = face_image[top:bottom, left:right]

        # Prepare image for gender prediction by resizing and subtracting mean values (normalizing pixels)
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Set input to gender detection model and perform forward pass
        self.gender_net.setInput(blob)
        gender_preds = self.gender_net.forward()

        # Get predicted gender category
        gender = self.gender_list[gender_preds[0].argmax()]

        return gender

    def estimate_age(self, face_image, face_location):
        # Extract face region from image
        top, right, bottom, left = face_location
        face = face_image[top:bottom, left:right]

        # Prepare image for age prediction
        blob = cv2.dnn.blobFromImage(face, 1, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)

        # Set input to age detection model and perform forward pass
        self.age_net.setInput(blob)
        age_preds = self.age_net.forward()

        # Get predicted age category
        age = self.age_list[age_preds[0].argmax()]

        return age

class FaceRecognitionGUI(QMainWindow):
    def __init__(self):
        super().__init__() #inheriting from the pyqt module called
        self.setWindowTitle("Face Recognition Statistics")
        self.setGeometry(100, 100, 800, 600)

        # Initialize face recognizer
        self.face_recognizer = FaceRecognizer()

        # Set up central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QHBoxLayout()
        central_widget.setLayout(layout)

        # Label for displaying video feed
        self.video_label = QLabel()
        layout.addWidget(self.video_label)

        # Layout for displaying statistics
        stats_layout = QVBoxLayout()
        layout.addLayout(stats_layout)

        # Label for displaying total unique faces count
        self.total_faces_label = QLabel("Total Unique Faces: 0")
        stats_layout.addWidget(self.total_faces_label)

        # Label for displaying current detected face information
        self.current_face_label = QLabel("Current Face: None")
        stats_layout.addWidget(self.current_face_label)

        # List widget for displaying recently detected faces
        stats_layout.addWidget(QLabel("Recently Detected Faces:"))
        self.recent_faces_list = QListWidget()
        stats_layout.addWidget(self.recent_faces_list)

        # Open video capture device
        # self.cap = cv2.VideoCapture(0)
        self.cap = cv2.VideoCapture('/home/bayanda/Downloads/vid.mp4')

        # Timer for updating video feed
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)  # Update every 30 ms

    def update_frame(self):
        # Read frame from video capture device
        ret, frame = self.cap.read()
        if ret:
            # Recognize faces in the frame
            face_locations, face_info = self.face_recognizer.recognize_face(frame)

            # Draw rectangles and text on the frame for each recognized face
            for (top, right, bottom, left), (name, gender, age) in zip(face_locations, face_info):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, f"{name}, {gender}, {age}", (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

            # Update current face label with information of the first recognized face
            if face_info:
                self.current_face_label.setText(f"Current Face: {face_info[0][0]}, {face_info[0][1]}, {face_info[0][2]}")
                self.update_recent_faces(f"{face_info[0][0]}, {face_info[0][1]}, {face_info[0][2]}")
            else:
                self.current_face_label.setText("Current Face: None")

            # Update total faces label with current unique face count
            self.total_faces_label.setText(f"Total Unique Faces: {self.face_recognizer.face_count}")

            # Convert frame to QImage and display in video label
            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            qt_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qt_image)
            self.video_label.setPixmap(pixmap)

    def update_recent_faces(self, face_info):
        # Insert recently detected face information at the top of the list
        self.recent_faces_list.insertItem(0, face_info)
        
        # Remove the oldest item if the list exceeds 10 items
        if self.recent_faces_list.count() > 10:
            self.recent_faces_list.takeItem(10)

    def closeEvent(self, event):
        # Release video capture device and save face database
        self.cap.release()
        self.face_recognizer.save_database()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = FaceRecognitionGUI()
    window.show()
    sys.exit(app.exec_())
