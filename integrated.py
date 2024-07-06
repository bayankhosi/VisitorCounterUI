import tkinter as tk
import pygame
import pygame.camera
from PIL import Image, ImageTk
import cv2
import mtcnn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque
import sqlite3
from datetime import datetime

class VisitorCounterApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Visitor Counter with Video Feed, Face Detection, and Analytics")
        
        self.visitor_count = 0
        self.visitor_history = deque(maxlen=100)
        
        pygame.init()
        pygame.camera.init()
        
        self.camera = pygame.camera.Camera(pygame.camera.list_cameras()[0], (640, 480))
        self.camera.start()
        
        self.detector = mtcnn.MTCNN()
        
        # Create a frame to hold video feed and analytics plot side by side
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()

        # Video feed canvas
        self.video_canvas = tk.Canvas(self.main_frame, width=640, height=480)
        self.video_canvas.grid(row=0, column=0, padx=10, pady=10)
        
        # Analytics plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title('Visitor Count Over Time')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Visitor Count')
        self.plot_line, = self.ax.plot([], [], label='Visitors')
        self.ax.legend()

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=1, padx=10, pady=10)
        
        # Visitor count label
        self.visitor_label = tk.Label(self.root, text=f"Current Visitors: {self.visitor_count}", font=("Helvetica", 24))
        self.visitor_label.pack(pady=20)
        
        # Buttons
        self.increment_button = tk.Button(self.root, text="Increment Visitor", command=self.increment_visitor)
        self.increment_button.pack(pady=10)
        
        self.reset_button = tk.Button(self.root, text="Reset Count", command=self.reset_visitor)
        self.reset_button.pack(pady=10)
        
        self.stream_video()
        self.update_analytics_plot()
        
        # Connect to SQLite database
        self.conn = sqlite3.connect('visitor_data.db')
        self.create_tables()

    def create_tables(self):
        # Create tables if they don't exist
        create_visitor_counts_table_sql = '''
        CREATE TABLE IF NOT EXISTS visitor_counts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            visitor_count INTEGER
        );
        '''
        create_faces_table_sql = '''
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            face_id TEXT NOT NULL,
            visitor_id INTEGER,
            FOREIGN KEY(visitor_id) REFERENCES visitors(id)
        );
        '''
        self.conn.execute(create_visitor_counts_table_sql)
        self.conn.execute(create_faces_table_sql)
        self.conn.commit()

    def stream_video(self):
        frame = self.camera.get_image()
        image_data = pygame.image.tostring(frame, "RGB")
        pil_image = Image.frombytes("RGB", (640, 480), image_data)
        
        faces = self.detector.detect_faces(np.array(pil_image))
        
        for face in faces:
            x, y, w, h = face['box']
            face_key = (x, y, w, h)
            
            existing_face_id = self.get_existing_face_id(face_key)
            
            if existing_face_id is None:
                visitor_id = self.increment_visitor()
                self.store_face_data(face_key, visitor_id)
                cv2.rectangle(np.array(pil_image), (x, y), (x+w, y+h), (0, 255, 0), 2)
            else:
                cv2.rectangle(np.array(pil_image), (x, y), (x+w, y+h), (255, 0, 0), 2)
        
        self.video_feed = ImageTk.PhotoImage(image=pil_image)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_feed)
        
        self.root.after(10, self.stream_video)
        
    def increment_visitor(self):
        self.visitor_count += 1
        self.visitor_history.append(self.visitor_count)
        self.update_visitor_label()
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        insert_visitor_count_sql = '''
        INSERT INTO visitor_counts (timestamp, visitor_count)
        VALUES (?, ?);
        '''
        self.conn.execute(insert_visitor_count_sql, (timestamp, self.visitor_count))
        self.conn.commit()
        
        return self.visitor_count
        
    def reset_visitor(self):
        self.visitor_count = 0
        self.visitor_history.clear()
        self.update_visitor_label()
        
        truncate_visitor_counts_sql = '''
        DELETE FROM visitor_counts;
        '''
        self.conn.execute(truncate_visitor_counts_sql)
        self.conn.commit()

        truncate_faces_sql = '''
        DELETE FROM faces;
        '''
        self.conn.execute(truncate_faces_sql)
        self.conn.commit()
        
    def update_visitor_label(self):
        self.visitor_label.config(text=f"Current Visitors: {self.visitor_count}")
        
    def update_analytics_plot(self):
        self.plot_line.set_data(range(len(self.visitor_history)), self.visitor_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        
        self.root.after(1000, self.update_analytics_plot)
        
    def get_existing_face_id(self, face_key):
        select_face_sql = '''
        SELECT id FROM faces WHERE face_id = ?
        '''
        face_id = self.conn.execute(select_face_sql, (str(face_key),)).fetchone()
        return face_id[0] if face_id else None
        
    def store_face_data(self, face_key, visitor_id):
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        insert_face_sql = '''
        INSERT INTO faces (timestamp, face_id, visitor_id)
        VALUES (?, ?, ?);
        '''
        self.conn.execute(insert_face_sql, (timestamp, str(face_key), visitor_id))
        self.conn.commit()
        
    def __del__(self):
        if hasattr(self, 'camera'):
            self.camera.stop()
        
        if self.conn:
            self.conn.close()

if __name__ == "__main__":
    root = tk.Tk()
    app = VisitorCounterApp(root)
    root.mainloop()
