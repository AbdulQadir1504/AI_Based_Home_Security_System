import os

# Video Settings
VIDEO_CAPTURE_DEVICE = 0  # For webcam (may not work in cloud)
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30

# Face Recognition
UNKNOWN_FACE_THRESHOLD = 0.6
FACE_MODEL = 'hog'  # Use 'hog' for speed in cloud
FACE_RECOGNITION_MODEL = 'VGG-Face'

# Alert Settings
ALERT_TRIGGER_COOLDOWN = 5  # seconds
MAX_ALERT_DISPLAY_TIME = 3  # seconds
ALERT_SOUND_ENABLED = False  # Disable sound in cloud

# Paths
KNOWN_FACES_DIR = 'known_faces'
ENCODINGS_FILE = 'face_encodings.pkl'
LOG_FILE = 'security_alerts.log'

# Create directories if they don't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
