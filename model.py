import numpy as np
import cv2
from scipy.signal import butter, filtfilt, hilbert
from scipy.fft import fft, fftfreq
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

# --- Auto-Load Face Detector ---
# We use the built-in OpenCV data path so you don't need to download XML manually.
cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(cascade_path)

def get_face_roi(frame):
    """
    Returns the face Region of Interest (ROI).
    If no face is detected, returns the center crop of the image.
    """
    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
        
        if len(faces) > 0:
            # Pick the largest face
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            # Return slightly tighter crop to avoid background
            return frame[y:y+h, x:x+w]
    except Exception:
        pass
        
    # Fallback: Center Crop (50% of image)
    h, w, _ = frame.shape
    start_x, start_y = int(w*0.25), int(h*0.25)
    end_x, end_y = int(w*0.75), int(h*0.75)
    return frame[start_y:end_y, start_x:end_x]

# ================= VIDEO LOGIC =================

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    if len(data) <= 15: return data
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data)

def compute_video_features(signal, fs=30):
    # Sanity check
    if len(signal) < fs or np.std(signal) < 1e-6:
        return [0, -20.0, 0]

    # Filter for human pulse range (0.7Hz - 3.5Hz)
    filtered = bandpass_filter(signal, 0.7, 3.5, fs)
    
    # Frequency Domain (FFT)
    N = len(filtered)
    yf = np.abs(fft(filtered))[:N//2]
    xf = fftfreq(N, 1/fs)[:N//2]
    
    # Find Dominant Frequency
    valid_band = np.where((xf >= 0.7) & (xf <= 3.5))
    if len(valid_band[0]) == 0: return [0, -20, 0]
    
    peak_idx_in_band = np.argmax(yf[valid_band])
    peak_idx = valid_band[0][peak_idx_in_band]
    dom_freq = xf[peak_idx]
    
    # SNR Calculation
    # Signal = Power around peak (+- 2 bins)
    # Noise = All other power
    win = 2
    idx_start = max(0, peak_idx - win)
    idx_end = min(len(yf), peak_idx + win + 1)
    
    signal_power = np.sum(yf[idx_start:idx_end]**2)
    total_power = np.sum(yf**2)
    noise_power = total_power - signal_power
    
    if noise_power < 1e-9: noise_power = 1e-9
    snr = 10 * np.log10(signal_power / noise_power)
    
    # Coherence (Phase Stability)
    try:
        analytic = hilbert(filtered)
        inst_phase = np.unwrap(np.angle(analytic))
        phase_diff = np.diff(inst_phase)
        # Low std dev = High Coherence
        coherence = 1.0 / (np.std(phase_diff) + 1e-5)
    except:
        coherence = 0.0

    return [dom_freq, snr, coherence]

def train_video_model():
    # Synthetic Data Training
    # Real: 0.8-2.5Hz, High SNR (>2dB), High Coherence (>1.5)
    # Fake: Random Freq, Low SNR (<0dB), Low Coherence (<1.0)
    X = []
    y = []
    for _ in range(300):
        # Class 1: Real
        X.append([np.random.uniform(0.8, 2.0), np.random.uniform(2.0, 15.0), np.random.uniform(1.5, 5.0)])
        y.append(1)
        # Class 0: Fake
        X.append([np.random.uniform(0.1, 4.0), np.random.uniform(-10.0, 1.0), np.random.uniform(0.0, 1.2)])
        y.append(0)
    
    clf = make_pipeline(StandardScaler(), SVC(probability=True, kernel='rbf'))
    clf.fit(X, y)
    return clf

# ================= PHOTO LOGIC =================

def compute_photo_features(image_bgr):
    try:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        
        # 1. Laplacian Variance (Texture/Grain)
        # Real photos have "good" noise. AI is often too smooth or weirdly sharp.
        lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # 2. FFT Mean (Pattern Detection)
        # Detects periodic noise (moiré patterns) from screens
        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = 20 * np.log(np.abs(fshift) + 1e-9)
        fft_mean = np.mean(magnitude)
        
        # 3. Color Stats
        (m_b, m_g, m_r), (s_b, s_g, s_r) = cv2.meanStdDev(image_bgr)
        color_std = np.mean([s_b, s_g, s_r])
        
        return [lap_var, fft_mean, color_std]
    except:
        return [0,0,0]

def train_photo_model():
    X = []
    y = []
    for _ in range(300):
        # Real: High Texture Var, Normal FFT, High Color Var
        X.append([np.random.uniform(100, 500), np.random.uniform(140, 160), np.random.uniform(40, 80)])
        y.append(1)
        # Fake (Smooth AI): Low Texture, Normal FFT, Low Color Var
        X.append([np.random.uniform(5, 50), np.random.uniform(140, 160), np.random.uniform(10, 40)])
        y.append(0)
        # Fake (Screen/Moiré): High FFT Mean
        X.append([np.random.uniform(50, 200), np.random.uniform(180, 250), np.random.uniform(20, 60)])
        y.append(0)
        
    clf = make_pipeline(StandardScaler(), SVC(probability=True))
    clf.fit(X, y)
    return clf