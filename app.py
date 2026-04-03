import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import sys

# Page configuration
st.set_page_config(
    page_title="AI Home Security System",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 AI Home Security System")
st.markdown("### Face Detection & Recognition System")

# Simple face detection function (works without dlib)
def simple_face_detection(image):
    """Simple face detection using OpenCV"""
    # Convert to grayscale
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    
    # Load OpenCV's face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(image, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    return image, len(faces)

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    st.success("✅ Application Running")
    
    # Show Python version
    st.code(f"Python: {sys.version_info.major}.{sys.version_info.minor}")
    
    st.markdown("---")
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Upload an image or video
    2. System will detect faces
    3. Green boxes show detected faces
    """)

# Main content
tab1, tab2 = st.tabs(["📸 Image Upload", "🎥 Video Upload"])

with tab1:
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file:
        # Read image
        image = Image.open(uploaded_file)
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        if len(image_np.shape) == 3:
            image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        else:
            image_bgr = image_np
        
        # Detect faces
        processed_image, face_count = simple_face_detection(image_bgr)
        
        # Convert back to RGB
        processed_image_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
        
        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Original Image", use_container_width=True)
        with col2:
            st.image(processed_image_rgb, caption=f"Detected {face_count} face(s)", use_container_width=True)
        
        st.success(f"✅ Found {face_count} face(s) in the image")

with tab2:
    uploaded_video = st.file_uploader("Choose a video", type=['mp4', 'avi', 'mov'])
    
    if uploaded_video:
        # Save video to temp file
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(uploaded_video.read())
        video_path = tfile.name
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        # Get video info
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Progress bar
        progress_bar = st.progress(0)
        video_placeholder = st.empty()
        
        frame_count = 0
        total_faces = 0
        
        while frame_count < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process every 5th frame for speed
            if frame_count % 5 == 0:
                processed_frame, face_count = simple_face_detection(frame)
                total_faces += face_count
                
                # Display frame
                processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                video_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
            
            # Update progress
            frame_count += 1
            progress_bar.progress(frame_count / total_frames)
        
        cap.release()
        os.unlink(video_path)
        
        st.success(f"✅ Processed {frame_count} frames, detected {total_faces} faces")

# Footer
st.markdown("---")
st.caption("AI Home Security System - Powered by OpenCV & Streamlit")
