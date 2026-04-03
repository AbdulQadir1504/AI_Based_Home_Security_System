# AI-Based Security System

An intelligent real-time surveillance system that detects humans, recognizes faces, and triggers intrusion alerts for unknown individuals.

## Features

✅ **Real-Time Face Detection** - Detects human faces using deep learning models  
✅ **Face Recognition** - Identifies known individuals from a face database  
✅ **Intrusion Alert System** - Triggers alerts when unknown persons are detected  
✅ **Alert Logging** - Records all security events with timestamps  
✅ **FPS Monitoring** - Tracks processing performance in real-time  
✅ **Screenshot Capture** - Save security footage snapshots on demand  
✅ **Windows Audio Alerts** - Sound notification for intrusion events  

## Project Structure

```
Final_Project/
├── app.py                 # Main application entry point
├── config.py              # System configuration settings
├── utils.py               # Face detection and encoding utilities
├── alert_system.py        # Alert and logging system
├── requirements.txt       # Python dependencies
├── known_faces/           # Directory for known person images
│   ├── PersonName1/
│   │   ├── image1.jpg
│   │   └── image2.jpg
│   └── PersonName2/
│       └── image1.jpg
├── security_alerts.log    # Alert event log (auto-generated)
└── README.md              # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- Webcam connected to your computer
- Windows OS (for audio alerts)

### Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Add Known Faces:**
   - Create subdirectories in `known_faces/` for each person:
     ```bash
     mkdir known_faces/John
     mkdir known_faces/Jane
     ```
   - Add at least 2-5 clear face photos per person (JPG, PNG, BMP, GIF)
   - Image requirements:
     - Clear frontal face view
     - Well-lit conditions
     - High resolution preferred
     - No extreme angles

## Usage

### Run the Original Desktop System

```bash
python app.py
```

### Run the FastAPI Backend + Streamlit Frontend

1. Start the FastAPI server:

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000 --reload
```

2. Start the Streamlit web dashboard:

```bash
streamlit run streamlit_app.py
```

3. Open the browser at the URL Streamlit prints (usually http://localhost:8501).

### Controls During Operation

| Key | Action |
|-----|--------|
| `q` | Quit the system |
| `s` | Save screenshot |
| `r` | Reset statistics |

### System Output

The system displays:
- **Real-time video stream** with face detection boxes
- **Person identification** (green box = known, red box = unknown)
- **Confidence scores** for face matching
- **FPS counter** showing processing speed
- **Frame and detection statistics**
- **Timestamp** of current capture
- **Alert overlay** for intrusion events

## Configuration

Edit `config.py` to customize:

### Video Settings
- `VIDEO_CAPTURE_DEVICE` - Webcam device (0 for default)
- `FRAME_WIDTH` / `FRAME_HEIGHT` - Resolution
- `FRAME_RATE` - Target FPS

### Face Recognition
- `UNKNOWN_FACE_THRESHOLD` - Sensitivity for unknown detection (0.6 default)
- `FACE_MODEL` - Detection model ('hog' for speed, 'cnn' for accuracy)

### Alert Settings
- `ALERT_TRIGGER_COOLDOWN` - Seconds between alerts for same person
- `MAX_ALERT_DISPLAY_TIME` - Alert display duration
- `ALERT_SOUND_ENABLED` - Enable/disable audio alerts

## Performance Tips

1. **Optimize Speed:**
   - Use `FACE_MODEL = "hog"` for faster detection
   - Reduce frame resolution for lighter processing
   - Close unnecessary applications

2. **Improve Accuracy:**
   - Use `FACE_MODEL = "cnn"` for better detection
   - Add multiple high-quality images per person (5-10)
   - Use clear, frontal images in good lighting
   - Increase `UNKNOWN_FACE_THRESHOLD` for stricter matching

3. **For Continuous Operation:**
   - Higher resolution images in the `known_faces/` directory
   - Balanced FPS settings (20-30 recommended)
   - Adequate system cooling for long-running sessions

## Alert System

### Intrusion Alert Triggering

When an **unknown person** is detected:
1. Visual alert displayed (red overlay)
2. Audio beep played (Windows)
3. Event logged to `security_alerts.log` with timestamp
4. Cooldown prevents duplicate alerts (default: 5 seconds)

### Alert Log Format

```
[2026-03-30 15:23:45] INTRUSION ALERT
  Status: UNKNOWN PERSON DETECTED
  Person ID: Unknown
  Confidence: 0.85
  Location: Entrance
  Action: Intrusion Alert Triggered
```

## Troubleshooting

### Issue: "No camera detected"
- Check webcam connection
- Try a different device number in `config.py`
- Test with other applications first

### Issue: Face detection is slow
- Switch to `FACE_MODEL = "hog"` (faster but less accurate)
- Reduce frame resolution
- Close background applications
- Check CPU usage

### Issue: Unknown face recognition is inaccurate
- Switch to `FACE_MODEL = "cnn"` for better accuracy
- Add more training images (5-10 per person)
- Ensure images are clear and well-lit
- Adjust `UNKNOWN_FACE_THRESHOLD` value

### Issue: No alerts triggering properly
- Check `ALERT_SOUND_ENABLED` setting
- Verify `ALERT_TRIGGER_COOLDOWN` isn't too high
- Ensure unknown faces have confidence > threshold
- Check volume levels (system and application)

## System Requirements

- **CPU:** Dual-core 2.0 GHz or higher
- **RAM:** 4GB minimum (8GB recommended)
- **Storage:** 200MB for application and data
- **Webcam:** USB webcam or integrated camera

## Deployment (FastAPI + Streamlit)

1. Use a production-grade ASGI server such as `uvicorn` or `gunicorn` with `uvicorn.workers.UvicornWorker`.

```bash
uvicorn api_server:app --host 0.0.0.0 --port 8000
```

2. Run Streamlit in server mode (or use native cloud deployment services):

```bash
streamlit run streamlit_app.py --server.headless true --server.port 8501
```

3. Reverse proxy (optional, recommended):
   - Use Nginx or Caddy to expose ports and provide HTTPS.
   - Route `/api` to `localhost:8000` and `/` to `localhost:8501`.

4. Docker example:

```Dockerfile
# Dockerfile for API
FROM python:3.11-slim
WORKDIR /app
COPY . /app
RUN pip install -r requirements.txt
EXPOSE 8000
CMD ["uvicorn", "api_server:app", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t ai-security-api .
docker run --rm -p 8000:8000 ai-security-api
```

5. For production resilience:
   - Use process manager (systemd, supervisord).
   - Add monitoring (Prometheus/Grafana).
   - Secure network with HTTPS and firewall rules.

## Advanced Usage

### Using Pre-Encoded Faces (Faster Loading)

First run:
```python
# This saves face encodings for faster subsequent runs
encoder = FaceEncoder()
encoder.load_known_faces()
encoder.save_encodings()
```

Subsequent runs will use the pickle file for instant loading.

### Custom Alert Handling

Modify `alert_system.py` to:
- Send email notifications
- Trigger HTTP webhooks
- Log to cloud services
- Integration with security systems

### Database Integration

Extend the system to:
- Store events in a database
- Generate reports and analytics
- Create timeline visualizations
- Export security data

## Security Notes

- This system is for legitimate security monitoring only
- Comply with all local privacy and surveillance laws
- Obtain proper consent from individuals being monitored
- Store and protect biometric data securely
- Regularly backup alert logs and recordings

## Performance Metrics

Typical performance on standard hardware:
- **Face Detection:** 20-30 FPS (HOG), 5-15 FPS (CNN)
- **Face Recognition:** Real-time (<50ms per face)
- **Memory Usage:** 200-500 MB
- **CPU Usage:** 20-40% (varies by model)

## Future Enhancements

- [ ] Multiple camera support
- [ ] Video recording capability
- [ ] Cloud integration for alerts
- [ ] Web dashboard for monitoring
- [ ] Advanced analytics and reporting
- [ ] Facial expression recognition
- [ ] Age and gender estimation
- [ ] Liveness detection (anti-spoofing)

## License

This project is provided as-is for security and surveillance purposes.

## Support

For issues or questions, check the Troubleshooting section or review the configuration settings in `config.py`.

---

**Version:** 1.0  
**Last Updated:** March 2026  
**Author:** AI Security System Development Team
