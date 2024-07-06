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

class VisitorCounterApp:
    def __init__(self, root):
        # Initialize the Tkinter root window
        self.root = root
        self.root.title("Visitor Counter with Video Feed, Face Detection, and Analytics")
        
        # Initialize visitor count and history
        self.visitor_count = 0
        self.visitor_history = deque(maxlen=100)  # Limit history to last 100 counts
        
        # Initialize Pygame and camera
        pygame.init()
        pygame.camera.init()
        
        # Initialize Pygame camera
        self.camera = pygame.camera.Camera(pygame.camera.list_cameras()[0], (640, 480))
        self.camera.start()
        
        # Initialize MTCNN detector
        self.detector = mtcnn.MTCNN()
        
        # Create a label to display visitor count
        self.visitor_label = tk.Label(self.root, text=f"Current Visitors: {self.visitor_count}", font=("Helvetica", 24))
        self.visitor_label.pack(pady=20)
        
        # Create a canvas to display video feed
        self.video_canvas = tk.Canvas(self.root, width=640, height=480)
        self.video_canvas.pack()
        
        # Create buttons for interaction
        self.increment_button = tk.Button(self.root, text="Increment Visitor", command=self.increment_visitor)
        self.increment_button.pack(pady=10)
        
        self.reset_button = tk.Button(self.root, text="Reset Count", command=self.reset_visitor)
        self.reset_button.pack(pady=10)
        
        # Initialize analytics plot using Matplotlib
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title('Visitor Count Over Time')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Visitor Count')
        self.plot_line, = self.ax.plot([], [], label='Visitors')
        self.ax.legend()
        
        # Create a canvas for Matplotlib plot and pack it into Tkinter window
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        
        # Start streaming video feed and updating analytics plot
        self.stream_video()
        self.update_analytics_plot()

    def stream_video(self):
        # Capture frame from camera using Pygame
        frame = self.camera.get_image()
        
        # Convert Pygame surface to PIL Image
        image_data = pygame.image.tostring(frame, "RGB")
        pil_image = Image.frombytes("RGB", (640, 480), image_data)
        
        # Detect faces using MTCNN
        faces = self.detector.detect_faces(np.array(pil_image))
        
        # Draw rectangles around detected faces
        for face in faces:
            x, y, w, h = face['box']
            cv2.rectangle(np.array(pil_image), (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Convert PIL Image to Tkinter Image format
        self.video_feed = ImageTk.PhotoImage(image=pil_image)
        
        # Update the video feed on the canvas
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_feed)
        
        # Repeat the function after 10 milliseconds for continuous streaming
        self.root.after(10, self.stream_video)
        
    def increment_visitor(self):
        # Increment visitor count and update history
        self.visitor_count += 1
        self.visitor_history.append(self.visitor_count)
        self.update_visitor_label()
        
    def reset_visitor(self):
        # Reset visitor count and clear history
        self.visitor_count = 0
        self.visitor_history.clear()
        self.update_visitor_label()
        
    def update_visitor_label(self):
        # Update the label displaying current visitor count
        self.visitor_label.config(text=f"Current Visitors: {self.visitor_count}")
        
    def update_analytics_plot(self):
        # Update the Matplotlib plot with visitor history data
        self.plot_line.set_data(range(len(self.visitor_history)), self.visitor_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        
        # Repeat the function every 1000 milliseconds (1 second) for analytics update
        self.root.after(1000, self.update_analytics_plot)
        
    def __del__(self):
        # Destructor to stop the camera when the object is deleted
        if hasattr(self, 'camera'):
            self.camera.stop()

if __name__ == "__main__":
    # Initialize Tkinter main window and start the application
    root = tk.Tk()
    app = VisitorCounterApp(root)
    root.mainloop()
