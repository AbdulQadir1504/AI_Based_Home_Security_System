import os

# Video Settings
VIDEO_CAPTURE_DEVICE = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30

# Face Recognition Settings
UNKNOWN_FACE_THRESHOLD = 0.6
FACE_MODEL = 'hog'  # 'hog' for speed, 'cnn' for accuracy
TOLERANCE = 0.6

# Alert Settings
ALERT_TRIGGER_COOLDOWN = 5  # seconds
MAX_ALERT_DISPLAY_TIME = 3  # seconds
ALERT_SOUND_ENABLED = False  # Disable sound for cloud

# Paths
KNOWN_FACES_DIR = 'known_faces'
ENCODINGS_FILE = 'face_encodings.pkl'
LOG_FILE = 'security_alerts.log'

# Create directories if they don't exist
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)
