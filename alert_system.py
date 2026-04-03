"""
Alert system for security events
"""

import time
from datetime import datetime
from pathlib import Path
from config import ALERT_LOG_FILE, ALERT_TRIGGER_COOLDOWN
import winsound


class AlertSystem:
    """Handles alerts and logging for security events"""
    
    def __init__(self):
        self.last_alert_time = {}
        self.alert_cooldown = ALERT_TRIGGER_COOLDOWN
        self.log_file = Path(ALERT_LOG_FILE)
        self.enable_sound = True
        self._initialize_log()
    
    def _initialize_log(self):
        """Initialize the alert log file"""
        if not self.log_file.exists():
            with open(self.log_file, 'w') as f:
                f.write("=" * 80 + "\n")
                f.write("AI SECURITY SYSTEM - ALERT LOG\n")
                f.write("=" * 80 + "\n\n")
    
    def can_trigger_alert(self, person_id):
        """Check if alert should be triggered (cooldown check)"""
        current_time = time.time()
        
        if person_id not in self.last_alert_time:
            self.last_alert_time[person_id] = current_time
            return True
        
        time_since_last_alert = current_time - self.last_alert_time[person_id]
        
        if time_since_last_alert >= self.alert_cooldown:
            self.last_alert_time[person_id] = current_time
            return True
        
        return False
    
    def trigger_alert(self, person_name, distance, location="Entrance"):
        """Trigger an intrusion alert for unknown person"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Log the alert
        log_message = (
            f"[{timestamp}] INTRUSION ALERT\n"
            f"  Status: UNKNOWN PERSON DETECTED\n"
            f"  Person ID: {person_name}\n"
            f"  Confidence: {distance:.2f}\n"
            f"  Location: {location}\n"
            f"  Action: Intrusion Alert Triggered\n"
            + "-" * 80 + "\n"
        )
        
        self._write_log(log_message)
        
        # Sound alert on Windows
        if self.enable_sound:
            self._play_sound_alert()
        
        return {
            'timestamp': timestamp,
            'person': person_name,
            'distance': distance,
            'location': location,
            'alert_type': 'INTRUSION'
        }
    
    def log_known_person(self, person_name, distance):
        """Log detection of known person"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        log_message = (
            f"[{timestamp}] KNOWN PERSON DETECTED\n"
            f"  Person: {person_name}\n"
            f"  Confidence: {distance:.2f}\n"
            + "-" * 80 + "\n"
        )
        
        self._write_log(log_message)
    
    def _play_sound_alert(self):
        """Play alert sound on Windows"""
        try:
            # Windows alert sound (frequency, duration in milliseconds)
            frequency = 1000  # 1000 Hz
            duration = 500    # 500 ms
            
            winsound.Beep(frequency, duration)
            # Play second beep for emphasis
            time.sleep(0.1)
            winsound.Beep(frequency, duration)
        except Exception as e:
            print(f"Error playing alert sound: {str(e)}")
    
    def _write_log(self, message):
        """Write message to log file"""
        try:
            with open(self.log_file, 'a') as f:
                f.write(message)
                f.write("\n")
        except Exception as e:
            print(f"Error writing to log file: {str(e)}")
    
    def get_log_summary(self):
        """Get summary of recent alerts"""
        try:
            with open(self.log_file, 'r') as f:
                content = f.read()
            
            intrusion_count = content.count("INTRUSION ALERT")
            return {
                'total_alerts': intrusion_count,
                'log_file': str(self.log_file)
            }
        except Exception as e:
            return {'error': str(e)}


def display_alert_on_frame(frame, alert_info, alert_display_time=3):
    """
    Display alert message on frame
    
    Args:
        frame: OpenCV frame
        alert_info: Dictionary with alert information
        alert_display_time: Time to display alert in seconds
    
    Returns:
        Modified frame with alert displayed
    """
    import cv2
    
    height, width = frame.shape[:2]
    
    # Red overlay for alert
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 255), -1)
    frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)
    
    # Alert text
    alert_text = "!!! INTRUSION ALERT - UNKNOWN PERSON DETECTED !!!"
    person_text = f"Person ID: {alert_info['person']}"
    confidence_text = f"Confidence: {alert_info['distance']:.2f}"
    
    cv2.putText(frame, alert_text, (10, 30), cv2.FONT_HERSHEY_BOLD, 1, (0, 0, 255), 2)
    cv2.putText(frame, person_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, confidence_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    return frame
