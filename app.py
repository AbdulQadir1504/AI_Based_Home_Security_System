"""
AI-Based Security System
Detects humans and recognizes faces using computer vision.
Identifies known/unknown persons and triggers alerts for intrusions.
"""

import cv2
import sys
import time
from pathlib import Path
from config import (
    VIDEO_CAPTURE_DEVICE, FRAME_RATE, FRAME_WIDTH, FRAME_HEIGHT,
    DISPLAY_CONFIDENCE, SHOW_FPS, WINDOW_TITLE,
    UNKNOWN_PERSON_COLOR, KNOWN_PERSON_COLOR,
    MAX_ALERT_DISPLAY_TIME
)
from utils import FaceEncoder, FaceDetector, create_known_faces_directory, get_system_stats
from alert_system import AlertSystem, display_alert_on_frame


class SecuritySystem:
    """Main AI-based security system"""
    
    def __init__(self):
        print("Initializing AI Security System...")
        
        # Ensure known_faces directory exists
        create_known_faces_directory()
        
        # Initialize components
        self.face_encoder = FaceEncoder()
        self.face_detector = FaceDetector()
        self.alert_system = AlertSystem()
        
        # Video capture
        self.cap = None
        self.is_running = False
        
        # FPS calculation
        self.fps = FRAME_RATE
        self.prev_frame_time = 0
        
        # Alert state
        self.current_alert = None
        self.alert_start_time = None
        
        # Statistics
        self.frames_processed = 0
        self.total_detections = 0
        self.unknown_detections = 0
        
        print("✓ System initialization complete")
    
    def load_known_faces(self):
        """Load known faces from directory"""
        print("\n" + "=" * 60)
        print("LOADING KNOWN FACES")
        print("=" * 60)
        
        success = self.face_encoder.load_known_faces()
        
        if success:
            stats = get_system_stats(self.face_encoder)
            print("\n✓ Successfully loaded known faces")
            print(f"  → Persons in database: {stats['known_persons']}")
            print(f"  → Total face encodings: {stats['loaded_faces']}")
        else:
            print("\n⚠ No known faces loaded. Add images to 'known_faces' directory:")
            print("  Structure: known_faces/PersonName/image1.jpg, image2.jpg, ...")
        
        print("=" * 60 + "\n")
        return success
    
    def initialize_video_capture(self):
        """Initialize video capture from webcam"""
        print("Initializing video capture...")
        
        try:
            self.cap = cv2.VideoCapture(VIDEO_CAPTURE_DEVICE)
            
            if not self.cap.isOpened():
                print("✗ Failed to open video capture device")
                return False
            
            # Set frame properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
            self.cap.set(cv2.CAP_PROP_FPS, FRAME_RATE)
            
            print(f"✓ Video capture initialized")
            print(f"  → Resolution: {FRAME_WIDTH}x{FRAME_HEIGHT}")
            print(f"  → Target FPS: {FRAME_RATE}")
            print(f"  → Device: {VIDEO_CAPTURE_DEVICE}\n")
            
            return True
        except Exception as e:
            print(f"✗ Error initializing video capture: {str(e)}")
            return False
    
    def process_frame(self, frame):
        """Process a single frame for face detection and recognition"""
        
        # Detect faces
        face_locations = self.face_detector.detect_faces(frame)
        
        if not face_locations:
            return frame, []
        
        # Get encodings for detected faces
        face_encodings = self.face_detector.get_face_encodings(frame, face_locations)
        
        detections = []
        
        # Process each detected face
        for face_location, face_encoding in zip(face_locations, face_encodings):
            name, distance, is_known = self.face_encoder.recognize_face(face_encoding)
            
            detection = {
                'name': name,
                'distance': distance,
                'is_known': is_known,
                'location': face_location
            }
            detections.append(detection)
            
            # Update statistics
            self.total_detections += 1
            if not is_known:
                self.unknown_detections += 1
            
            # Determine color
            color = KNOWN_PERSON_COLOR if is_known else UNKNOWN_PERSON_COLOR
            
            # Draw on frame
            frame = self.face_detector.draw_box_and_label(
                frame, face_location, name, distance, is_known, color,
                show_distance=DISPLAY_CONFIDENCE
            )
            
            # Handle unknown person alert
            if not is_known and self.alert_system.can_trigger_alert(name):
                self.current_alert = self.alert_system.trigger_alert(name, distance)
                self.alert_start_time = time.time()
                print(f"\n🚨 INTRUSION ALERT: Unknown person detected (confidence: {distance:.2f})")
            
            # Log known person
            if is_known:
                self.alert_system.log_known_person(name, distance)
        
        return frame, detections
    
    def draw_system_info(self, frame):
        """Draw system information on frame"""
        height, width = frame.shape[:2]
        
        # FPS calculation
        current_time = time.time()
        if self.prev_frame_time == 0:
            self.prev_frame_time = current_time
        
        fps = 1 / (current_time - self.prev_frame_time)
        self.prev_frame_time = current_time
        
        # Draw info
        info_lines = [
            f"Frames: {self.frames_processed}",
            f"Total Detections: {self.total_detections}",
            f"Unknown Detections: {self.unknown_detections}"
        ]
        
        if SHOW_FPS:
            info_lines.insert(0, f"FPS: {fps:.1f}")
        
        y_pos = 30
        for info in info_lines:
            cv2.putText(frame, info, (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y_pos += 25
        
        # Draw timestamp
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        return frame
    
    def run(self):
        """Main security system loop"""
        
        print("\n" + "=" * 60)
        print("STARTING AI SECURITY SYSTEM")
        print("=" * 60)
        print("\nControls:")
        print("  - Press 'q' to quit")
        print("  - Press 's' to save screenshot")
        print("  - Press 'r' to reset statistics")
        print("\n" + "=" * 60 + "\n")
        
        # Load known faces
        if not self.load_known_faces():
            print("⚠ Warning: No known faces loaded. System will only detect UNKNOWN persons.\n")
        
        # Initialize video capture
        if not self.initialize_video_capture():
            print("✗ Failed to initialize video capture. Exiting.")
            return False
        
        self.is_running = True
        
        try:
            while self.is_running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("✗ Failed to read frame from video capture")
                    break
                
                self.frames_processed += 1
                
                # Process frame
                frame, detections = self.process_frame(frame)
                
                # Draw system info
                frame = self.draw_system_info(frame)
                
                # Display alert if active
                if self.current_alert:
                    time_since_alert = time.time() - self.alert_start_time
                    if time_since_alert < MAX_ALERT_DISPLAY_TIME:
                        frame = display_alert_on_frame(frame, self.current_alert)
                    else:
                        self.current_alert = None
                
                # Display frame
                cv2.imshow(WINDOW_TITLE, frame)
                
                # Handle key presses
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q'):
                    print("\nShutting down system...")
                    break
                elif key == ord('s'):
                    self._save_screenshot(frame)
                elif key == ord('r'):
                    self._reset_statistics()
        
        except KeyboardInterrupt:
            print("\n\nInterrupt signal received. Shutting down...")
        except Exception as e:
            print(f"\n✗ Error during execution: {str(e)}")
        finally:
            self.cleanup()
        
        return True
    
    def _save_screenshot(self, frame):
        """Save a screenshot of current frame"""
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        filename = f"security_screenshot_{timestamp}.jpg"
        
        try:
            cv2.imwrite(filename, frame)
            print(f"✓ Screenshot saved: {filename}")
        except Exception as e:
            print(f"✗ Error saving screenshot: {str(e)}")
    
    def _reset_statistics(self):
        """Reset system statistics"""
        self.frames_processed = 0
        self.total_detections = 0
        self.unknown_detections = 0
        print("✓ Statistics reset")
    
    def print_final_report(self):
        """Print final system report"""
        print("\n" + "=" * 60)
        print("SECURITY SYSTEM SHUTDOWN REPORT")
        print("=" * 60)
        
        alert_summary = self.alert_system.get_log_summary()
        
        print(f"\nSession Statistics:")
        print(f"  → Frames Processed: {self.frames_processed}")
        print(f"  → Total Face Detections: {self.total_detections}")
        print(f"  → Unknown Person Detections: {self.unknown_detections}")
        
        if self.total_detections > 0:
            unknown_percentage = (self.unknown_detections / self.total_detections) * 100
            print(f"  → Unknown Detection Rate: {unknown_percentage:.1f}%")
        
        print(f"\nSecurity Events:")
        print(f"  → Total Intrusion Alerts: {alert_summary.get('total_alerts', 0)}")
        print(f"  → Alert Log: {alert_summary.get('log_file', 'N/A')}")
        
        print("\n✓ System shutdown complete. All data logged.")
        print("=" * 60 + "\n")
    
    def cleanup(self):
        """Clean up resources"""
        self.is_running = False
        
        if self.cap:
            self.cap.release()
        
        cv2.destroyAllWindows()
        
        self.print_final_report()


def main():
    """Main entry point"""
    try:
        system = SecuritySystem()
        system.run()
    except Exception as e:
        print(f"\n✗ Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
