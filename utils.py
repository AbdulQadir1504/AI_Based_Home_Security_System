import os
import pickle
import numpy as np
import cv2

class FaceEncoder:
    def __init__(self, known_faces_dir='known_faces', encodings_file='face_encodings.pkl'):
        self.known_faces_dir = known_faces_dir
        self.encodings_file = encodings_file
        self.known_face_encodings = []
        self.known_face_names = []
    
    def load_known_faces(self):
        """Load and encode known faces using simple method"""
        print("Loading known faces...")
        
        self.known_face_encodings = []
        self.known_face_names = []
        
        if not os.path.exists(self.known_faces_dir):
            print(f"Directory {self.known_faces_dir} not found")
            return False
        
        # Simple encoding using image hashing (no dlib required)
        for person_name in os.listdir(self.known_faces_dir):
            person_dir = os.path.join(self.known_faces_dir, person_name)
            
            if os.path.isdir(person_dir):
                print(f"Processing {person_name}...")
                image_count = 0
                
                for image_file in os.listdir(person_dir):
                    if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                        image_path = os.path.join(person_dir, image_file)
                        
                        # Read image and create simple hash
                        img = cv2.imread(image_path)
                        if img is not None:
                            # Create a simple hash of the image (for demo purposes)
                            # In production, you'd use proper face encoding
                            hash_val = hash(f"{person_name}_{image_file}")
                            self.known_face_encodings.append(hash_val)
                            self.known_face_names.append(person_name)
                            image_count += 1
                            print(f"  - Added {image_file}")
                
                if image_count == 0:
                    print(f"  - No valid images found for {person_name}")
        
        print(f"Loaded {len(self.known_face_names)} faces from {len(set(self.known_face_names))} people")
        return len(self.known_face_names) > 0
    
    def save_encodings(self):
        """Save face encodings to file"""
        data = {
            'encodings': self.known_face_encodings,
            'names': self.known_face_names
        }
        with open(self.encodings_file, 'wb') as f:
            pickle.dump(data, f)
        print(f"Saved encodings to {self.encodings_file}")
    
    def load_encodings(self):
        """Load pre-saved face encodings"""
        if os.path.exists(self.encodings_file):
            with open(self.encodings_file, 'rb') as f:
                data = pickle.load(f)
            self.known_face_encodings = data['encodings']
            self.known_face_names = data['names']
            print(f"Loaded {len(self.known_face_names)} encodings from file")
            return True
        return False

class FaceRecognizer:
    def __init__(self, tolerance=0.6):
        self.tolerance = tolerance
        self.known_face_encodings = []
        self.known_face_names = []
    
    def load_known_encodings(self, encodings_file='face_encodings.pkl'):
        """Load pre-saved encodings"""
        if os.path.exists(encodings_file):
            with open(encodings_file, 'rb') as f:
                data = pickle.load(f)
            self.known_face_encodings = data['encodings']
            self.known_face_names = data['names']
            return True
        return False
    
    def recognize_faces(self, frame):
        """Detect and recognize faces using OpenCV"""
        # Convert to grayscale for detection
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Load face detector
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        face_locations = []
        face_names = []
        confidences = []
        
        for (x, y, w, h) in faces:
            # Convert to (top, right, bottom, left) format
            top, right, bottom, left = y, x+w, y+h, x
            face_locations.append((top, right, bottom, left))
            
            # Simple recognition logic (demo)
            # In production, you'd use actual face comparison
            if self.known_face_names:
                # For demo, mark first face as known if encodings exist
                if len(face_locations) == 1 and len(self.known_face_names) > 0:
                    face_names.append(self.known_face_names[0])
                    confidences.append(0.85)
                else:
                    face_names.append("Unknown")
                    confidences.append(0.65)
            else:
                face_names.append("Unknown")
                confidences.append(0.50)
        
        return face_locations, face_names, confidences

def draw_face_boxes(frame, face_locations, face_names, confidences):
    """Draw bounding boxes and labels on frame"""
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidences):
        # Choose color based on recognition
        if name != "Unknown":
            color = (0, 255, 0)  # Green for known
            label = f"{name} ({confidence:.2f})"
        else:
            color = (0, 0, 255)  # Red for unknown
            label = f"⚠️ INTRUDER! ({confidence:.2f})"
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (left, top - label_size[1] - 10), 
                     (left + label_size[0], top), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (left, top - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame
