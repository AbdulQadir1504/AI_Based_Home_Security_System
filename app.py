import streamlit as st
import cv2
import numpy as np
from PIL import Image

# MUST be the first Streamlit command
st.set_page_config(
    page_title="AI Home Security",
    page_icon="🏠",
    layout="wide"
)

# Title
st.title("🏠 AI Home Security System")
st.markdown("### Face Detection System")

# Simple face detection function
def detect_faces(uploaded_image):
    # Convert to OpenCV format
    image = Image.open(uploaded_image)
    image_np = np.array(image)
    
    # Convert RGB to BGR
    if len(image_np.shape) == 3:
        image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    else:
        image_bgr = image_np
    
    # Convert to grayscale
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Draw rectangles
    for (x, y, w, h) in faces:
        cv2.rectangle(image_bgr, (x, y), (x+w, y+h), (0, 255, 0), 3)
        cv2.putText(image_bgr, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Convert back to RGB
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    return image_rgb, len(faces)

# Sidebar
with st.sidebar:
    st.header("📊 System Status")
    st.success("✅ System Running")
    st.info("Upload an image to detect faces")
    
    st.markdown("---")
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. Upload an image
    2. System detects faces
    3. Green boxes show detections
    """)

# Main content
st.header("📸 Upload Image for Face Detection")

uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=['jpg', 'jpeg', 'png'],
    help="Upload a clear image with faces"
)

if uploaded_file is not None:
    # Display original and processed
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(uploaded_file, use_container_width=True)
    
    with col2:
        st.subheader("Detection Result")
        with st.spinner("Detecting faces..."):
            processed_image, face_count = detect_faces(uploaded_file)
            st.image(processed_image, use_container_width=True)
    
    # Show results
    if face_count > 0:
        st.success(f"✅ **Found {face_count} face(s)** in the image")
        
        # Alert for multiple faces
        if face_count > 3:
            st.warning("⚠️ Multiple faces detected - Security alert!")
    else:
        st.info("ℹ️ No faces detected. Try another image with clear faces.")
    
    # Add download button
        st.download_button(
            label="📸 Download Processed Image",
            data=cv2.imencode('.jpg', cv2.cvtColor(processed_image, cv2.COLOR_RGB2BGR))[1].tobytes(),
            file_name="detected_faces.jpg",
            mime="image/jpeg"
        )

# Footer
st.markdown("---")
st.caption("AI Home Security System | Powered by OpenCV & Streamlit")

# Show that app is alive
st.markdown("### 🟢 System Status: Active")
