import cv2
import numpy as np
import requests
from flask import Flask, Response, render_template, request
from threading import Thread, Lock
import time
import urllib.parse
import imutils

app = Flask(__name__)

# Load models
face_net = cv2.dnn.readNetFromCaffe(
    "deploy.prototxt", 
    "res10_300x300_ssd_iter_140000.caffemodel"
)
plate_cascade = cv2.CascadeClassifier("haarcascade_russian_plate_number.xml")

# Global variables for camera configuration
camera_config = {
    'ip': '192.168.1.100',  # Default IP
    'port': '8080',         # Default port
    'path': 'video',        # Default path
    'username': '',
    'password': '',
    'is_active': False
}

# Frame processing lock
frame_lock = Lock()
current_frame = None
processing_frame = None

# Global settings
blur_settings = {
    'face_blur': True,
    'plate_blur': True,
    'blur_intensity': 99
}

def generate_camera_url():
    """Generate the camera stream URL based on configuration"""
    auth_part = ""
    if camera_config['username'] and camera_config['password']:
        auth_part = f"{camera_config['username']}:{camera_config['password']}@"
    
    return f"http://{auth_part}{camera_config['ip']}:{camera_config['port']}/{camera_config['path']}"

def process_frame(frame):
    """Apply face and license plate blurring to a frame"""
    if frame is None:
        return None
        
    # Create a copy for plate detection
    plate_detection_frame = frame.copy()
    h, w = frame.shape[:2]
    
    # Face detection and blurring
    if blur_settings['face_blur']:
        blob = cv2.dnn.blobFromImage(
            cv2.resize(frame, (300, 300)), 
            1.0, 
            (300, 300), 
            (104.0, 177.0, 123.0)
        )
        face_net.setInput(blob)
        detections = face_net.forward()
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure coordinates are within frame
                startX, startY = max(0, startX), max(0, startY)
                endX, endY = min(w, endX), min(h, endY)
                
                if endY > startY and endX > startX:
                    face_roi = frame[startY:endY, startX:endX]
                    # Adjust blur intensity based on setting
                    k = blur_settings['blur_intensity'] // 2 * 2 + 1
                    frame[startY:endY, startX:endX] = cv2.GaussianBlur(
                        face_roi, (k, k), 0
                    )
    
    # License plate detection and blurring
    if blur_settings['plate_blur']:
        gray = cv2.cvtColor(plate_detection_frame, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in plates:
            # Apply blur to plate
            plate_roi = frame[y:y+h, x:x+w]
            k = blur_settings['blur_intensity'] // 2 * 2 + 1
            frame[y:y+h, x:x+w] = cv2.GaussianBlur(plate_roi, (k, k), 0)
    
    return frame

def capture_frames():
    """Capture frames from IP camera and process them"""
    global current_frame, processing_frame
    
    while True:
        if not camera_config['is_active']:
            time.sleep(1)
            continue
            
        try:
            # Generate camera URL
            camera_url = generate_camera_url()
            
            # Capture frame from IP camera
            stream = requests.get(camera_url, stream=True, timeout=5)
            if stream.status_code != 200:
                print(f"Failed to connect to camera: {stream.status_code}")
                time.sleep(2)
                continue
                
            bytes_data = bytes()
            for chunk in stream.iter_content(chunk_size=1024):
                bytes_data += chunk
                a = bytes_data.find(b'\xff\xd8')  # JPEG start
                b = bytes_data.find(b'\xff\xd9')  # JPEG end
                
                if a != -1 and b != -1:
                    jpg = bytes_data[a:b+2]
                    bytes_data = bytes_data[b+2:]
                    
                    # Decode the JPEG image
                    frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                    
                    if frame is not None:
                        # Resize for processing
                        frame = imutils.resize(frame, width=800)
                        
                        # Process frame
                        processed_frame = process_frame(frame.copy())
                        
                        # Update frames
                        with frame_lock:
                            current_frame = frame.copy()
                            processing_frame = processed_frame
        except Exception as e:
            print(f"Camera error: {str(e)}")
            time.sleep(2)

# Start the frame capture thread
capture_thread = Thread(target=capture_frames, daemon=True)
capture_thread.start()

def generate_frames(processed=True):
    """Generate video frames for streaming"""
    while True:
        with frame_lock:
            if processed and processing_frame is not None:
                frame = processing_frame.copy()
            elif not processed and current_frame is not None:
                frame = current_frame.copy()
            else:
                # Create a placeholder frame
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "No camera feed", (150, 240), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Convert frame to JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
        
        # Small delay to reduce CPU load
        time.sleep(0.03)

# Video feed routes
@app.route('/video_feed')
def video_feed():
    """Processed video feed with blurring"""
    return Response(generate_frames(processed=True), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/raw_video_feed')
def raw_video_feed():
    """Raw video feed without processing"""
    return Response(generate_frames(processed=False), 
                    mimetype='multipart/x-mixed-replace; boundary=frame')

# Update settings route
@app.route('/update_settings', methods=['POST'])
def update_settings():
    global blur_settings
    data = request.json
    blur_settings['face_blur'] = data.get('face_blur', True)
    blur_settings['plate_blur'] = data.get('plate_blur', True)
    blur_settings['blur_intensity'] = int(data.get('blur_intensity', 99))
    return '', 204

# Camera configuration route
@app.route('/configure_camera', methods=['POST'])
def configure_camera():
    global camera_config
    camera_config['ip'] = request.form.get('ip', '192.168.1.100')
    camera_config['port'] = request.form.get('port', '8080')
    camera_config['path'] = request.form.get('path', 'video')
    camera_config['username'] = request.form.get('username', '')
    camera_config['password'] = request.form.get('password', '')
    camera_config['is_active'] = True
    return '', 204

# Main page
@app.route('/')
def index():
    return render_template('templates\index1.html', 
                           settings=blur_settings,
                           config=camera_config)

def generate_camera_url():
    """Generate the camera stream URL based on configuration"""
    auth_part = ""
    if camera_config['username'] and camera_config['password']:
        auth_part = f"{camera_config['username']}:{camera_config['password']}@"
    
    return f"http://{auth_part}{camera_config['ip']}:{camera_config['port']}/{camera_config['path']}"

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)