import tkinter as tk
import pygame
import pygame.camera
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
from collections import deque
import threading
import time

class VisitorCounterApp:
    """
    A GUI application to count visitors, display a video feed, and show analytics using a histogram.
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
        
        # Create a main frame to hold video feed and analytics plot side by side
        self.main_frame = tk.Frame(self.root)
        self.main_frame.pack()
        
        # Create label to display current visitor count
        self.visitor_label = tk.Label(self.main_frame, text=f"Current Visitors: {self.visitor_count}", font=("Helvetica", 24))
        self.visitor_label.grid(row=0, column=0, padx=10, pady=10)
        
        # Create canvas for video feed
        self.video_canvas = tk.Canvas(self.main_frame, width=640, height=480)
        self.video_canvas.grid(row=1, column=0, padx=10, pady=10)
        
        # Create button to increment visitor count
        self.increment_button = tk.Button(self.main_frame, text="Increment Visitor", command=self.increment_visitor)
        self.increment_button.grid(row=2, column=0, pady=10)
        
        # Create button to reset visitor count
        self.reset_button = tk.Button(self.main_frame, text="Reset Count", command=self.reset_visitor)
        self.reset_button.grid(row=3, column=0, pady=10)
        
        # Initialize pygame and pygame camera
        pygame.init()
        pygame.camera.init()
        
        # Initialize camera and start capturing
        self.camera = pygame.camera.Camera(pygame.camera.list_cameras()[0], (640, 480))
        self.camera.start()
        
        # Initialize video streaming thread
        self.video_thread = threading.Thread(target=self.stream_video)
        self.video_thread.daemon = True  # Set as daemon to stop with main thread
        self.video_thread.start()
        
        # Initialize matplotlib figure and axes for analytics plot
        self.fig, self.ax = plt.subplots(figsize=(8, 4))
        self.ax.set_title('Visitor Count Distribution')
        self.ax.set_xlabel('Visitor Count')
        self.ax.set_ylabel('Frequency')
        
        # Initialize the bins for the histogram
        self.visitor_bins = np.arange(0, 101, 5)  # Adjust as needed
        
        # Initialize the histogram bars
        self.visitor_hist = self.ax.bar(self.visitor_bins[:-1], np.zeros(len(self.visitor_bins)-1), width=5)

        # Set y-axis limits
        self.ax.set_ylim(0, 20)

        # Embed matplotlib plot into tkinter GUI
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.main_frame)
        self.canvas.get_tk_widget().grid(row=0, column=1, rowspan=4, padx=10, pady=10)
        
        # Start updating analytics plot periodically
        self.update_analytics_plot()

    def stream_video(self):
        """
        Continuously updates the video feed from the camera.

        Uses pygame to capture frames from the camera, converts them to
        a tkinter-compatible format, and updates the video canvas.
        """
        while True:
            frame = self.camera.get_image()
            
            # Convert pygame surface to PIL image
            image_data = pygame.image.tostring(frame, "RGB")
            pil_image = Image.frombytes("RGB", (640, 480), image_data)
            
            # Update video canvas with the new frame
            self.video_feed = ImageTk.PhotoImage(image=pil_image)
            self.video_canvas.create_image(0, 0, anchor=tk.NW, image=self.video_feed)
            
            # Sleep for a short time to prevent excessive CPU usage
            time.sleep(0.03)  # Adjust as needed

    def increment_visitor(self):
        """
        Increment the visitor count by 1, update the visitor history,
        and trigger an immediate update of the analytics plot.
        """
        self.visitor_count += 1
        self.visitor_history.append(self.visitor_count)
        self.update_visitor_label()
        self.update_analytics_plot()  # Call update_analytics_plot here

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
        Update the analytics plot with the latest visitor history data, 
        setting the x-axis intervals to represent visitor count bins.
        """
        if not self.visitor_history:  # Check if visitor_history is empty
            return  # Do nothing if empty

        # Update the histogram data
        visitor_counts, _ = np.histogram(self.visitor_history, bins=self.visitor_bins)

        # Update histogram bars heights
        for rect, count in zip(self.visitor_hist, visitor_counts):
            rect.set_height(count)

        # Redraw plot
        self.canvas.draw()

        # Schedule next update of analytics plot after 1 minute (60000 milliseconds)
        self.root.after(60000, self.update_analytics_plot)
        
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
