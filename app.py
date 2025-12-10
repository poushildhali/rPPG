import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify
from model import train_video_model, train_photo_model, compute_video_features, compute_photo_features, get_face_roi

app = Flask(__name__)

# --- Configuration ---
UPLOAD_FOLDER = 'temp_uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# --- Load Models on Startup ---
print("Training/Loading Video Model...")
video_clf = train_video_model()
print("Training/Loading Photo Model...")
photo_clf = train_photo_model()
print("System Ready.")

ALLOWED_VIDEO = {'mp4', 'avi', 'mov', 'mkv'}
ALLOWED_PHOTO = {'jpg', 'jpeg', 'png', 'webp'}

def get_file_type(filename):
    if '.' not in filename: return None
    ext = filename.rsplit('.', 1)[1].lower()
    if ext in ALLOWED_VIDEO: return 'video'
    if ext in ALLOWED_PHOTO: return 'photo'
    return None

# --- Video Logic Wrapper ---
def process_video_signal(video_path):
    cap = cv2.VideoCapture(video_path)
    signal = []
    frame_count = 0
    
    if not cap.isOpened(): return None

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame_count += 1
        # Process every frame for accuracy (can skip if slow)
        
        # 1. Face Detection & ROI Extraction
        roi = get_face_roi(frame)
        
        # 2. Extract Green Channel (Strongest PPG signal)
        # ROI is (Height, Width, BGR)
        green_mean = np.mean(roi[:, :, 1])
        signal.append(green_mean)
        
        # Safety limit (15 seconds max to prevent server hang)
        if len(signal) > 450: break
    
    cap.release()
    
    # 3. Validation
    signal = np.array(signal)
    if len(signal) < 60: # Need at least ~2 seconds
        return None
        
    # Detrend (Remove DC offset)
    return signal - np.mean(signal)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    file_type = get_file_type(file.filename)
    if not file_type:
        return jsonify({'error': 'Unsupported file format'}), 400

    # Save file securely
    temp_path = os.path.join(UPLOAD_FOLDER, f"temp_{file.filename}")
    file.save(temp_path)

    try:
        if file_type == 'video':
            # --- VIDEO PIPELINE ---
            raw_signal = process_video_signal(temp_path)
            
            if raw_signal is None:
                raise ValueError("Could not detect a stable face/heartbeat. Ensure video is >3 seconds and face is visible.")
            
            features = compute_video_features(raw_signal, fs=30)
            prediction = video_clf.predict([features])[0]
            probs = video_clf.predict_proba([features])[0]
            confidence = probs[prediction] * 100

            result_text = "REAL HUMAN" if prediction == 1 else "FAKE / SPOOF"
            
            return jsonify({
                'type': 'video',
                'result': result_text,
                'confidence': f"{confidence:.1f}%",
                'stats': {
                    'Heart Rate Freq': f"{features[0]:.2f} Hz",
                    'Signal Quality (SNR)': f"{features[1]:.2f} dB",
                    'Pulse Stability': f"{features[2]:.3f}"
                }
            })

        elif file_type == 'photo':
            # --- PHOTO PIPELINE ---
            image = cv2.imread(temp_path)
            if image is None:
                raise ValueError("Could not read image.")
            
            # (Optional) Check if face exists in photo before analyzing
            # face_roi = get_face_roi(image) 
            # If you want to analyze only face texture, use face_roi here instead of image.
            
            features = compute_photo_features(image)
            prediction = photo_clf.predict([features])[0]
            probs = photo_clf.predict_proba([features])[0]
            confidence = probs[prediction] * 100

            result_text = "REAL PHOTO" if prediction == 1 else "AI / SCREEN"

            return jsonify({
                'type': 'photo',
                'result': result_text,
                'confidence': f"{confidence:.1f}%",
                'stats': {
                    'Texture Detail': f"{features[0]:.1f}",
                    'Pattern Artifacts': f"{features[1]:.1f}",
                    'Color Naturalness': f"{features[2]:.1f}"
                }
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500
        
    finally:
        # Cleanup
        if os.path.exists(temp_path):
            os.remove(temp_path)

if __name__ == '__main__':
    app.run(debug=True)