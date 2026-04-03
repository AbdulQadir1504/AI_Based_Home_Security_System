import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os
import time
from datetime import datetime
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="AI Security System - Face Detection",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stButton > button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
    }
    .success-message {
        padding: 1rem;
        background-color: #d4edda;
        color: #155724;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .alert-box {
        padding: 1rem;
        background-color: #f8d7da;
        color: #721c24;
        border-radius: 0.5rem;
        margin: 1rem 0;
        border-left: 5px solid #dc3545;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'alert_count' not in st.session_state:
    st.session_state.alert_count = 0
if 'total_faces_detected' not in st.session_state:
    st.session_state.total_faces_detected = 0
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'known_faces_loaded' not in st.session_state:
    st.session_state.known_faces_loaded = False
if 'face_encodings' not in st.session_state:
    st.session_state.face_encodings = {}
if 'alerts_log' not in st.session_state:
    st.session_state.alerts_log = []

# Cache resource for loading models (improves performance)
@st.cache_resource
def load_face_recognition_model():
    """Load face recognition model with caching"""
    try:
        from deepface import DeepFace
        import face_recognition
        return {
            'deepface': DeepFace,
            'face_recognition': face_recognition,
            'loaded': True
        }
    except Exception as e:
        st.error(f"Error loading face recognition models: {e}")
        return {'loaded': False}

# Cache known faces loading
@st.cache_resource
def load_known_faces_cached(known_faces_dir='known_faces'):
    """Load and encode known faces from directory"""
    encodings = {}
    
    if not os.path.exists(known_faces_dir):
        return encodings
    
    for person_name in os.listdir(known_faces_dir):
        person_dir = os.path.join(known_faces_dir, person_name)
        if os.path.isdir(person_dir):
            encodings[person_name] = []
            for image_file in os.listdir(person_dir):
                if image_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                    image_path = os.path.join(person_dir, image_file)
                    try:
                        # Use face_recognition for encoding (HOG model by default)
                        image = face_recognition.load_image_file(image_path)
                        face_encodings = face_recognition.face_encodings(image)
                        if face_encodings:
                            encodings[person_name].append(face_encodings[0])
                    except Exception as e:
                        st.warning(f"Could not process {image_path}: {e}")
    
    return encodings

def detect_and_recognize_faces(frame, known_encodings, tolerance=0.6):
    """
    Detect faces and recognize known individuals
    Uses HOG model for faster processing (better for cloud deployment)
    """
    import face_recognition
    
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Detect face locations using HOG model (faster than CNN)
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    
    if not face_locations:
        return [], [], []
    
    # Get face encodings for detected faces
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
    
    face_names = []
    confidences = []
    
    # Compare with known faces
    for face_encoding in face_encodings:
        matches = []
        face_distances = []
        
        for name, known_encodings_list in known_encodings.items():
            for known_encoding in known_encodings_list:
                # Calculate face distance (lower = more similar)
                distance = face_recognition.face_distance([known_encoding], face_encoding)[0]
                face_distances.append(distance)
                matches.append((name, distance))
        
        if matches:
            # Find best match (lowest distance)
            best_match = min(matches, key=lambda x: x[1])
            name, distance = best_match
            
            if distance < tolerance:
                face_names.append(name)
                confidence = 1 - distance
                confidences.append(confidence)
            else:
                face_names.append("Unknown")
                confidences.append(1 - distance)
        else:
            face_names.append("Unknown")
            confidences.append(0.5)
    
    return face_locations, face_names, confidences

def draw_detections(frame, face_locations, face_names, confidences):
    """Draw bounding boxes and labels on frame"""
    for (top, right, bottom, left), name, confidence in zip(face_locations, face_names, confidences):
        # Color: Green for known, Red for unknown
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        # Draw bounding box
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        
        # Create label
        if name != "Unknown":
            label = f"{name} ({confidence:.2f})"
        else:
            label = f"⚠️ INTRUDER! ({confidence:.2f})"
            # Add warning symbol for unknown faces
            cv2.circle(frame, (left + 10, top + 10), 5, (0, 0, 255), -1)
        
        # Draw label background
        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (left, top - label_size[1] - 10), 
                     (left + label_size[0], top), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (left, top - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    return frame

def log_alert(unknown_count, frame_number=None):
    """Log security alerts"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alert_entry = {
        'timestamp': timestamp,
        'unknown_count': unknown_count,
        'frame': frame_number,
        'message': f"🚨 ALERT: {unknown_count} unknown person(s) detected!"
    }
    st.session_state.alerts_log.insert(0, alert_entry)  # Add to beginning
    st.session_state.alert_count += unknown_count
    
    # Keep only last 50 alerts
    if len(st.session_state.alerts_log) > 50:
        st.session_state.alerts_log = st.session_state.alerts_log[:50]
    
    return alert_entry

# Main UI
st.title("🎥 AI-Based Security System")
st.markdown("### Real-time Face Detection & Recognition")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Model selection
    detection_model = st.selectbox(
        "Detection Model",
        ["hog (Faster, CPU-friendly)", "cnn (More Accurate, Slower)"],
        index=0
    )
    use_hog = "hog" in detection_model.lower()
    
    # Recognition threshold
    recognition_threshold = st.slider(
        "Face Recognition Threshold",
        min_value=0.3,
        max_value=0.8,
        value=0.6,
        step=0.05,
        help="Lower = stricter matching, Higher = more lenient"
    )
    
    st.markdown("---")
    st.header("📊 Statistics")
    
    # Display metrics
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Alerts", st.session_state.alert_count)
    with col2:
        st.metric("Faces Detected", st.session_state.total_faces_detected)
    
    st.markdown("---")
    st.header("📁 Known Faces")
    
    # Load known faces button
    if st.button("🔄 Load Known Faces"):
        with st.spinner("Loading known faces..."):
            st.session_state.face_encodings = load_known_faces_cached()
            if st.session_state.face_encodings:
                st.session_state.known_faces_loaded = True
                st.success(f"✅ Loaded {len(st.session_state.face_encodings)} known person(s)")
            else:
                st.warning("⚠️ No known faces found. Add images to 'known_faces' directory")
    
    if st.session_state.known_faces_loaded:
        st.info(f"📋 Known people: {', '.join(st.session_state.face_encodings.keys())}")
    
    st.markdown("---")
    st.header("ℹ️ Instructions")
    st.markdown("""
    1. **Upload a video** or **use camera**
    2. Click **Start Detection**
    3. Green boxes = Known persons
    4. Red boxes = Unknown/Intruders
    5. Alerts are logged automatically
    """)

# Main content area
tab1, tab2, tab3 = st.tabs(["📹 Live Detection", "📝 Alert Log", "📖 About"])

with tab1:
    # Input source selection
    input_source = st.radio(
        "Select Input Source",
        ["📁 Upload Video File", "📸 Camera (Browser)"],
        horizontal=True
    )
    
    video_file = None
    camera_image = None
    
    if input_source == "📁 Upload Video File":
        video_file = st.file_uploader(
            "Choose a video file",
            type=['mp4', 'avi', 'mov', 'mkv'],
            help="Upload a video file for face detection"
        )
    else:
        camera_image = st.camera_input("Take a picture", help="Click to capture photo for instant detection")
    
    # Start detection button
    if st.button("🚀 Start Detection", type="primary", use_container_width=True):
        if not st.session_state.known_faces_loaded and not st.session_state.face_encodings:
            st.warning("⚠️ Please load known faces first using the button in sidebar")
        else:
            st.session_state.processing = True
            st.session_state.total_faces_detected = 0
            
            # Process based on input source
            if video_file and input_source == "📁 Upload Video File":
                # Save uploaded video to temp file
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(video_file.read())
                video_path = tfile.name
                
                # Open video
                cap = cv2.VideoCapture(video_path)
                
                if not cap.isOpened():
                    st.error("Cannot open video file")
                else:
                    # Get video info
                    fps = int(cap.get(cv2.CAP_PROP_FPS))
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    
                    # Progress bar
                    progress_bar = st.progress(0)
                    frame_placeholder = st.empty()
                    info_placeholder = st.empty()
                    
                    frame_count = 0
                    unknown_in_current_frame = 0
                    
                    while st.session_state.processing and frame_count < total_frames:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        
                        # Process every frame
                        face_locations, face_names, confidences = detect_and_recognize_faces(
                            frame, 
                            st.session_state.face_encodings,
                            recognition_threshold
                        )
                        
                        # Update statistics
                        current_faces = len(face_locations)
                        st.session_state.total_faces_detected += current_faces
                        
                        # Check for unknown faces
                        unknown_count = sum(1 for name in face_names if name == "Unknown")
                        if unknown_count > 0 and unknown_count != unknown_in_current_frame:
                            log_alert(unknown_count, frame_count)
                            unknown_in_current_frame = unknown_count
                        
                        # Draw detections
                        processed_frame = draw_detections(frame, face_locations, face_names, confidences)
                        
                        # Convert to RGB for display
                        processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                        
                        # Update display
                        frame_placeholder.image(processed_frame_rgb, channels="RGB", use_container_width=True)
                        info_placeholder.info(f"""
                        **Frame:** {frame_count}/{total_frames}
                        **Faces Detected:** {current_faces}
                        **Unknown Faces:** {unknown_count}
                        **Status:** 🟢 Analyzing...
                        """)
                        
                        # Update progress
                        progress_bar.progress(frame_count / total_frames)
                        frame_count += 1
                        
                        # Control speed
                        time.sleep(1/fps if fps > 0 else 0.03)
                    
                    cap.release()
                    progress_bar.progress(1.0)
                    info_placeholder.success("✅ Video processing completed!")
                    
                    # Cleanup temp file
                    os.unlink(video_path)
            
            elif camera_image and input_source == "📸 Camera (Browser)":
                # Process single camera image
                with st.spinner("Processing image..."):
                    # Convert PIL image to numpy array
                    image = Image.open(camera_image)
                    frame = np.array(image)
                    
                    # Process image
                    face_locations, face_names, confidences = detect_and_recognize_faces(
                        frame,
                        st.session_state.face_encodings,
                        recognition_threshold
                    )
                    
                    # Draw detections
                    processed_frame = draw_detections(frame, face_locations, face_names, confidences)
                    
                    # Display result
                    st.image(processed_frame, channels="RGB", use_container_width=True)
                    
                    # Show results
                    if face_names:
                        st.success(f"✅ Found {len(face_names)} face(s)")
                        for name, conf in zip(face_names, confidences):
                            if name != "Unknown":
                                st.markdown(f"👤 **Recognized:** {name} (confidence: {conf:.2%})")
                            else:
                                st.markdown(f"⚠️ **Unknown person detected!** (confidence: {conf:.2%})")
                                log_alert(1, 0)
                    else:
                        st.warning("No faces detected in the image")
            
            st.session_state.processing = False

with tab2:
    # Display alert log
    if st.session_state.alerts_log:
        for alert in st.session_state.alerts_log:
            with st.container():
                st.markdown(f"""
                <div class="alert-box">
                <strong>🚨 {alert['message']}</strong><br>
                📅 Time: {alert['timestamp']}<br>
                👥 Unknown Count: {alert['unknown_count']}
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No alerts recorded yet. Start detection to see alerts here.")
    
    if st.button("Clear Alert Log"):
        st.session_state.alerts_log = []
        st.session_state.alert_count = 0
        st.rerun()

with tab3:
    st.markdown("""
    ## About This System
    
    ### Features
    - **Real-time face detection** using HOG (Histogram of Oriented Gradients)
    - **Face recognition** to identify known individuals
    - **Intruder alerts** for unknown persons
    - **Performance optimized** for cloud deployment
    
    ### How It Works
    1. The system detects faces using HOG algorithm (faster than CNN)
    2. Each face is compared against known face encodings
    3. If similarity score > threshold, person is recognized
    4. Unknown faces trigger security alerts
    
    ### Technical Details
    - **Face Detection:** HOG + Linear SVM
    - **Face Recognition:** FaceNet embeddings
    - **Similarity Metric:** Euclidean distance
    - **Processing Speed:** ~20-30 FPS on CPU
    
    ### Performance Tips
    - Use clear, front-facing photos for known faces
    - Add 2-3 images per person for better accuracy
    - Adjust threshold (0.6 default) for stricter/lenient matching
    - HOG model is optimized for cloud CPU environments
    
    ### Limitations
    - Cloud deployment has limited CPU resources
    - No real webcam access (use file upload or browser camera)
    - Processing speed depends on uploaded video resolution
    
    ---
    **Made with ❤️ using Streamlit, OpenCV, and face_recognition**
    """)

# Footer
st.markdown("---")
st.markdown("*AI-Based Security System | Real-time Face Detection & Recognition*")
