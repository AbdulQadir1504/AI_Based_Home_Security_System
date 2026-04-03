import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
from utils import FaceEncoder, FaceRecognizer
from alert_system import AlertSystem
import time

# Page config
st.set_page_config(
    page_title="AI Security System",
    page_icon="🎥",
    layout="wide"
)

# Initialize session state
if 'running' not in st.session_state:
    st.session_state.running = False
if 'alert_system' not in st.session_state:
    st.session_state.alert_system = AlertSystem()
if 'face_encoder' not in st.session_state:
    st.session_state.face_encoder = FaceEncoder()
    st.session_state.face_encoder.load_known_faces()
if 'frame_placeholder' not in st.session_state:
    st.session_state.frame_placeholder = None

st.title("🎥 AI-Based Security System")
st.markdown("---")

# Sidebar controls
with st.sidebar:
    st.header("Controls")
    
    # Camera source selection
    source_type = st.radio("Select Input Source", ["Webcam", "Upload Video"])
    
    if source_type == "Upload Video":
        uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    start_button = st.button("Start Detection", type="primary")
    stop_button = st.button("Stop Detection")
    
    st.markdown("---")
    st.header("Statistics")
    alert_stats = st.session_state.alert_system.get_statistics()
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Alerts", alert_stats.get('total_alerts', 0))
    with col2:
        st.metric("Unknown Persons", alert_stats.get('unknown_count', 0))
    
    st.markdown("---")
    st.header("Recent Alerts")
    recent_alerts = st.session_state.alert_system.get_recent_alerts(5)
    for alert in recent_alerts:
        st.text(alert)

# Main display area
col1, col2 = st.columns([3, 1])

with col1:
    frame_placeholder = st.empty()
    
with col2:
    st.subheader("Detection Info")
    info_placeholder = st.empty()

# Function to process frames
def process_frame(frame, recognizer, alert_system):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations, face_names, confidences = recognizer.recognize_faces(rgb_frame)
    
    # Draw results
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidences):
        # Color: Green for known, Red for unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # Draw box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Label
        label = f"{name} ({confidence:.2f})" if name != "Unknown" else f"INTRUDER! ({confidence:.2f})"
        cv2.putText(frame, label, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Trigger alert for unknown
        if name == "Unknown" and confidence > 0.6:
            alert_system.trigger_alert(confidence)
    
    return frame, len(face_locations)

# Main detection loop
if start_button:
    st.session_state.running = True
    
    recognizer = FaceRecognizer()
    recognizer.load_known_encodings()
    
    # Initialize video source
    cap = None
    if source_type == "Webcam":
        cap = cv2.VideoCapture(0)
    elif source_type == "Upload Video" and uploaded_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        cap = cv2.VideoCapture(tfile.name)
    
    if cap and not cap.isOpened():
        st.error("Cannot open video source")
        st.session_state.running = False
    else:
        while st.session_state.running:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            processed_frame, num_faces = process_frame(
                frame, recognizer, st.session_state.alert_system
            )
            
            # Convert to RGB for display
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Update displays
            frame_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
            info_placeholder.info(f"👤 Faces Detected: {num_faces}\n⏱️ Status: Active")
            
            time.sleep(0.03)  # ~30 FPS
        
        cap.release()

if stop_button:
    st.session_state.running = False
    info_placeholder.info("Detection Stopped")
