"""
Configuration settings for the AI Security System
"""

# Video Settings
VIDEO_CAPTURE_DEVICE = 0  # Default webcam
FRAME_RATE = 30
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Face Recognition Settings
KNOWN_FACES_DIR = "known_faces"
UNKNOWN_FACE_THRESHOLD = 0.6  # Distance threshold for face recognition
# PROFESSIONAL TOGGLE: 
# Use "hog" for real-time performance on CPUs (Standard Laptops).
# Use "cnn" for maximum accuracy if an NVIDIA GPU is available.
# During the demo, we use "hog" to ensure a smooth 30 FPS experience.
FACE_MODEL = "hog"
# Alert Settings
ALERT_TRIGGER_COOLDOWN = 5  # Seconds between alerts for same person
MAX_ALERT_DISPLAY_TIME = 3  # Seconds to display alert on screen
ALERT_SOUND_ENABLED = True
ALERT_LOG_FILE = "security_alerts.log"

# Display Settings
DISPLAY_CONFIDENCE = True
SHOW_FPS = True
WINDOW_TITLE = "AI Security System - Real-Time Face Recognition"
UNKNOWN_PERSON_COLOR = (0, 0, 255)  # Red for unknown
KNOWN_PERSON_COLOR = (0, 255, 0)   # Green for known
ALERT_COLOR = (0, 0, 255)           # Red for alert

# Encoding Settings
FACE_ENCODING_MODEL = "small"  # 'small' is faster, suitable for real-time
