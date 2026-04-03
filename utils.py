import os
import cv2
import face_recognition
import numpy as np
from pathlib import Path
import pickle
from config import KNOWN_FACES_DIR, FACE_MODEL, UNKNOWN_FACE_THRESHOLD

class FaceEncoder:
    def __init__(self):
        self.known_face_encodings = []
        self.known_face_names = []
        self.encoding_model = FACE_MODEL
        
    def load_known_faces(self):
        faces_dir = Path(KNOWN_FACES_DIR)
        if not faces_dir.exists():
            faces_dir.mkdir(parents=True, exist_ok=True) # Professional: Create if missing
            return False
        
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        
        for person_dir in faces_dir.iterdir():
            if not person_dir.is_dir():
                continue
                
            person_name = person_dir.name
            
            for image_file in person_dir.iterdir():
                if image_file.suffix.lower() not in valid_extensions:
                    continue
                
                try:
                    # FIX: Use face_recognition's built-in loader which handles RGB correctly
                    image = face_recognition.load_image_file(str(image_file))
                    
                    # Professional: Use 'hog' for CPU or 'cnn' for GPU via config
                    encodings = face_recognition.face_encodings(image, model=self.encoding_model)
                    
                    if len(encodings) > 0:
                        self.known_face_encodings.append(encodings[0])
                        self.known_face_names.append(person_name)
                except Exception as e:
                    print(f"Error loading {image_file}: {e}")
        
        return len(self.known_face_encodings) > 0

    def recognize_face(self, face_encoding):
        if not self.known_face_encodings:
            return "Unknown", 1.0, False
        
        # Calculate Euclidean distance
        distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
        best_match_index = np.argmin(distances)
        distance = distances[best_match_index]
        
        if distance <= UNKNOWN_FACE_THRESHOLD:
            return self.known_face_names[best_match_index], distance, True
        return "Unknown", distance, False

class FaceDetector:
    def __init__(self, model=FACE_MODEL):
        self.model = model
    
    def detect_faces(self, frame):
        """Optimized detection."""
        # FIX: Ensure we are working with a 3-channel BGR image
        if frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Professional: Return locations to be used by the encoder
        return face_recognition.face_locations(rgb_frame, model=self.model)
    
    def get_face_encodings(self, frame, face_locations):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return face_recognition.face_encodings(rgb_frame, face_locations)
    
    @staticmethod
    def draw_box_and_label(frame, face_location, name, distance, is_known, color, show_distance=True):
        top, right, bottom, left = face_location
        
        # Draw the main box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Professional Label Styling
        label = f"{name}" + (f" ({distance:.2f})" if show_distance else "")
        if not is_known: label = "ALARM: UNKNOWN"

        # Draw a filled rectangle for the text background (makes it readable)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), color, cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, label, (left + 6, bottom - 6), font, 0.6, (255, 255, 255), 1)
        
        return frame

def create_known_faces_directory():
    Path(KNOWN_FACES_DIR).mkdir(parents=True, exist_ok=True)
    
def get_system_stats():
    """Returns a basic dictionary of system status for the standalone app."""
    return {
        "Status": "Running",
        "Engine": FACE_MODEL.upper(),
        "Threshold": UNKNOWN_FACE_THRESHOLD
    }