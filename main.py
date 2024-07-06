import tkinter as tk
import pygame
import pygame.camera
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import deque

class VisitorCounterApp:
    """
    A GUI application to count visitors, display a video feed, and show analytics.

    This application integrates tkinter for the GUI, pygame for video streaming,
    and matplotlib for analytics plotting.
    """

    def __init__(self, root):
        """
        Initialize the VisitorCounterApp instance.

        Parameters:
        - root: The tkinter root window.
        """
        # Initialize tkinter root window
        self.root = root
        self.root.title("Visitor Counter with Video Feed and Analytics")
        
        # Initialize visitor count and history
        self.visitor_count = 0
        self.visitor_history = deque(maxlen=100)
        
        # Create label to display current visitor count
        self.visitor_label = tk.Label(self.root, text=f"Current Visitors: {self.visitor_count}", font=("Helvetica", 24))
        self.visitor_label.pack(pady=20)
        
        # Create canvas for video feed
        self.video_canvas = tk.Canvas(self.root, width=640, height=480)
        self.video_canvas.pack()
        
        # Create button to increment visitor count
        self.increment_button = tk.Button(self.root, text="Increment Visitor", command=self.increment_visitor)
        self.increment_button.pack(pady=10)
        
        # Create button to reset visitor count
        self.reset_button = tk.Button(self.root, text="Reset Count", command=self.reset_visitor)
        self.reset_button.pack(pady=10)
        
        # Initialize pygame and pygame camera
        pygame.init()
        pygame.camera.init()
        
        # Initialize camera and start capturing
        self.camera = pygame.camera.Camera(pygame.camera.list_cameras()[0], (640, 480))
        self.camera.start()
        
        # Start streaming video feed
        self.stream_video()
        
        # Initialize matplotlib figure and axes for analytics plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title('Visitor Count Over Time')
        self.ax.set_xlabel('Time')
        self.ax.set_ylabel('Visitor Count')
        self.plot_line, = self.ax.plot([], [], label='Visitors')
        self.ax.legend()
        
        # Embed matplotlib plot into tkinter GUI
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack()
        
        # Start updating analytics plot periodically
        self.update_analytics_plot()

    def stream_video(self):
        """
        Continuously updates the video feed from the camera.

        Uses pygame to capture frames from the camera, converts them to
        a tkinter-compatible format, and updates the video canvas.
        """
        frame = self.camera.get_image()
        
        # Convert pygame surface to PIL image
        image_data = pygame.image.tostring(frame, "RGB")
        pil_image = Image.frombytes("RGB", (640, 480), image_data)
        
        # Update video canvas with the new frame
        self.video_feed = ImageTk.PhotoImage(image=pil_image)
        self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_feed)
        
        # Schedule next video frame update
        self.root.after(10, self.stream_video)
        
    def increment_visitor(self):
        """
        Increment the visitor count by 1 and update the visitor history.
        Has to done automatically once integrated with face detection
        """
        self.visitor_count += 1
        self.visitor_history.append(self.visitor_count)
        self.update_visitor_label()
        
    def reset_visitor(self):
        """
        Reset the visitor count to 0 and clear the visitor history.
        """
        self.visitor_count = 0
        self.visitor_history.clear()
        self.update_visitor_label()
        
    def update_visitor_label(self):
        """
        Update the visitor count label with the current visitor count.
        """
        self.visitor_label.config(text=f"Current Visitors: {self.visitor_count}")
        
    def update_analytics_plot(self):
        """
        Update the analytics plot with the latest visitor history data.
        """
        self.plot_line.set_data(range(len(self.visitor_history)), self.visitor_history)
        self.ax.relim()
        self.ax.autoscale_view()
        self.fig.canvas.draw()
        
        # Schedule next update of analytics plot
        self.root.after(1000, self.update_analytics_plot)
        
    def __del__(self):
        """
        Destructor to ensure the camera is stopped when the object is deleted.
        """
        if hasattr(self, 'camera'):
            self.camera.stop()

if __name__ == "__main__":
    # Create the tkinter root window and start the application
    root = tk.Tk()
    app = VisitorCounterApp(root)
    root.mainloop()
