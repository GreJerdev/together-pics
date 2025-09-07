import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import mediapipe as mp
import os
import shutil
from PIL import Image, ImageTk
from pathlib import Path
import threading
import numpy as np

class PersonImageClassifier:
    def __init__(self, root):
        self.root = root
        self.root.title("Person Image Classifier")
        self.root.geometry("1000x700")
        
        # Initialize MediaPipe
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)
        self.face_mesh = self.mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=10, min_detection_confidence=0.5)
        
        # Variables
        self.selected_folder = tk.StringVar()
        self.target_person_landmarks = None
        self.target_person_face_crop = None
        self.current_image_path = None
        self.image_files = []
        self.current_image_index = 0
        
        # Create GUI
        self.create_widgets()
        
    def create_widgets(self):
        # Main frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        main_frame.rowconfigure(2, weight=1)
        
        # Folder selection
        ttk.Label(main_frame, text="Select Image Folder:").grid(row=0, column=0, sticky=tk.W, pady=5)
        ttk.Entry(main_frame, textvariable=self.selected_folder, width=50).grid(row=0, column=1, sticky=(tk.W, tk.E), padx=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_folder).grid(row=0, column=2, padx=5)
        
        # Control buttons
        control_frame = ttk.Frame(main_frame)
        control_frame.grid(row=1, column=0, columnspan=3, pady=10)
        
        ttk.Button(control_frame, text="Load Images", command=self.load_images).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Previous", command=self.previous_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Next", command=self.next_image).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Select Person", command=self.select_person).pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Start Classification", command=self.start_classification).pack(side=tk.LEFT, padx=5)
        
        # Image display
        self.image_label = ttk.Label(main_frame, text="No image loaded", anchor="center")
        self.image_label.grid(row=2, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=10)
        
        # Status
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(main_frame, textvariable=self.status_var)
        status_label.grid(row=3, column=0, columnspan=3, pady=5)
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(main_frame, variable=self.progress_var, maximum=100)
        self.progress_bar.grid(row=4, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
    def detect_faces(self, image_path):
        """Detect faces in an image and return face locations and crops"""
        image = cv2.imread(image_path)
        if image is None:
            return [], []
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        face_locations = []
        face_crops = []
        
        if results.detections:
            h, w, _ = image.shape
            for detection in results.detections:
                bbox = detection.location_data.relative_bounding_box
                x = int(bbox.xmin * w)
                y = int(bbox.ymin * h)
                width = int(bbox.width * w)
                height = int(bbox.height * h)
                
                # Ensure coordinates are within image bounds
                x = max(0, x)
                y = max(0, y)
                width = min(width, w - x)
                height = min(height, h - y)
                
                face_locations.append((y, x + width, y + height, x))
                face_crop = image[y:y+height, x:x+width]
                face_crops.append(face_crop)
        
        return face_locations, face_crops
    
    def extract_face_landmarks(self, face_crop):
        """Extract facial landmarks from a face crop"""
        if face_crop is None or face_crop.size == 0:
            return None
        
        rgb_face = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_face)
        
        if results.multi_face_landmarks:
            # Return the first face's landmarks
            landmarks = results.multi_face_landmarks[0]
            # Convert to numpy array for easier comparison
            landmark_points = []
            for landmark in landmarks.landmark:
                landmark_points.append([landmark.x, landmark.y, landmark.z])
            return np.array(landmark_points)
        
        return None
    
    def compare_faces(self, landmarks1, landmarks2, threshold=0.1):
        """Compare two sets of facial landmarks"""
        if landmarks1 is None or landmarks2 is None:
            return False
        
        if len(landmarks1) != len(landmarks2):
            return False
        
        # Calculate the mean squared error between landmarks
        mse = np.mean((landmarks1 - landmarks2) ** 2)
        return mse < threshold
        
    def browse_folder(self):
        folder = filedialog.askdirectory()
        if folder:
            self.selected_folder.set(folder)
            
    def load_images(self):
        folder = self.selected_folder.get()
        if not folder:
            messagebox.showerror("Error", "Please select a folder first")
            return
            
        # Get all image files
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        self.image_files = []
        
        for file_path in Path(folder).rglob('*'):
            if file_path.suffix.lower() in image_extensions:
                self.image_files.append(str(file_path))
                
        if not self.image_files:
            messagebox.showwarning("Warning", "No image files found in the selected folder")
            return
            
        self.current_image_index = 0
        self.display_current_image()
        self.status_var.set(f"Loaded {len(self.image_files)} images")
        
    def display_current_image(self):
        if not self.image_files or self.current_image_index >= len(self.image_files):
            return
            
        self.current_image_path = self.image_files[self.current_image_index]
        
        try:
            # Load and resize image for display
            image = Image.open(self.current_image_path)
            image.thumbnail((800, 600), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(image)
            self.image_label.configure(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
            
            # Update status
            self.status_var.set(f"Image {self.current_image_index + 1} of {len(self.image_files)}: {os.path.basename(self.current_image_path)}")
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not load image: {e}")
            
    def previous_image(self):
        if self.image_files and self.current_image_index > 0:
            self.current_image_index -= 1
            self.display_current_image()
            
    def next_image(self):
        if self.image_files and self.current_image_index < len(self.image_files) - 1:
            self.current_image_index += 1
            self.display_current_image()
            
    def select_person(self):
        if not self.current_image_path:
            messagebox.showerror("Error", "Please load and select an image first")
            return
            
        try:
            # Detect faces in the current image
            face_locations, face_crops = self.detect_faces(self.current_image_path)
            
            if not face_locations:
                messagebox.showwarning("Warning", "No faces detected in this image")
                return
                
            if len(face_locations) == 1:
                # Only one face, use it
                face_crop = face_crops[0]
                landmarks = self.extract_face_landmarks(face_crop)
                if landmarks is not None:
                    self.target_person_landmarks = landmarks
                    self.target_person_face_crop = face_crop
                    messagebox.showinfo("Success", "Person selected successfully!")
                else:
                    messagebox.showerror("Error", "Could not extract facial features from the selected face")
            else:
                # Multiple faces, let user choose
                self.choose_face_from_multiple(face_locations, face_crops)
                
        except Exception as e:
            messagebox.showerror("Error", f"Could not process image: {e}")
            
    def choose_face_from_multiple(self, face_locations, face_crops):
        # Create a new window for face selection
        face_window = tk.Toplevel(self.root)
        face_window.title("Select Person")
        face_window.geometry("600x400")
        
        # Load the original image for display
        image = cv2.imread(self.current_image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Draw rectangles around faces
        for i, (top, right, bottom, left) in enumerate(face_locations):
            cv2.rectangle(rgb_image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(rgb_image, f"Face {i+1}", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        pil_image.thumbnail((500, 300), Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(pil_image)
        
        # Display image
        image_label = ttk.Label(face_window, image=photo)
        image_label.pack(pady=10)
        image_label.image = photo
        
        # Face selection buttons
        button_frame = ttk.Frame(face_window)
        button_frame.pack(pady=10)
        
        for i in range(len(face_crops)):
            ttk.Button(button_frame, text=f"Select Face {i+1}", 
                      command=lambda idx=i: self.select_face_and_close(face_crops[idx], face_window)).pack(side=tk.LEFT, padx=5)
                      
    def select_face_and_close(self, face_crop, window):
        landmarks = self.extract_face_landmarks(face_crop)
        if landmarks is not None:
            self.target_person_landmarks = landmarks
            self.target_person_face_crop = face_crop
            window.destroy()
            messagebox.showinfo("Success", "Person selected successfully!")
        else:
            messagebox.showerror("Error", "Could not extract facial features from the selected face")
        
    def start_classification(self):
        if self.target_person_landmarks is None:
            messagebox.showerror("Error", "Please select a person first")
            return
            
        if not self.image_files:
            messagebox.showerror("Error", "Please load images first")
            return
            
        # Create output folders
        folder = self.selected_folder.get()
        person_folder = os.path.join(folder, "a_person")
        not_person_folder = os.path.join(folder, "not_person")
        
        os.makedirs(person_folder, exist_ok=True)
        os.makedirs(not_person_folder, exist_ok=True)
        
        # Start classification in a separate thread
        self.progress_var.set(0)
        self.status_var.set("Starting classification...")
        
        thread = threading.Thread(target=self.classify_images, args=(person_folder, not_person_folder))
        thread.daemon = True
        thread.start()
        
    def classify_images(self, person_folder, not_person_folder):
        try:
            total_images = len(self.image_files)
            person_count = 0
            not_person_count = 0
            error_count = 0
            
            for i, image_path in enumerate(self.image_files):
                try:
                    # Detect faces in the image
                    face_locations, face_crops = self.detect_faces(image_path)
                    
                    person_found = False
                    if face_crops:
                        # Compare each detected face with the target person
                        for face_crop in face_crops:
                            landmarks = self.extract_face_landmarks(face_crop)
                            if landmarks is not None:
                                if self.compare_faces(self.target_person_landmarks, landmarks):
                                    person_found = True
                                    break
                            
                    # Copy to appropriate folder
                    filename = os.path.basename(image_path)
                    # Handle duplicate filenames by adding a counter
                    base_name, ext = os.path.splitext(filename)
                    counter = 1
                    target_folder = person_folder if person_found else not_person_folder
                    final_filename = filename
                    
                    while os.path.exists(os.path.join(target_folder, final_filename)):
                        final_filename = f"{base_name}_{counter}{ext}"
                        counter += 1
                    
                    shutil.copy2(image_path, os.path.join(target_folder, final_filename))
                    
                    if person_found:
                        person_count += 1
                    else:
                        not_person_count += 1
                        
                except Exception as e:
                    print(f"Error processing {image_path}: {e}")
                    error_count += 1
                    
                # Update progress
                progress = (i + 1) / total_images * 100
                self.progress_var.set(progress)
                self.status_var.set(f"Processing {i+1}/{total_images}: {os.path.basename(image_path)}")
                
            # Classification complete
            result_message = f"Classification complete!\nPerson images: {person_count}\nNot person images: {not_person_count}"
            if error_count > 0:
                result_message += f"\nErrors: {error_count}"
            
            self.status_var.set(f"Classification complete! Person: {person_count}, Not Person: {not_person_count}")
            messagebox.showinfo("Complete", result_message)
            
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("Error", f"Classification failed: {e}")
    
    def cleanup(self):
        """Clean up MediaPipe resources"""
        if hasattr(self, 'face_detection'):
            self.face_detection.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

def main():
    root = tk.Tk()
    app = PersonImageClassifier(root)
    
    def on_closing():
        app.cleanup()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()
