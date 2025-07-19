import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from config import (
    LIVENESS_HISTORY_LENGTH, LIVENESS_EAR_STD_THRESHOLD, LIVENESS_HEAD_MOVE_STD_THRESHOLD,
    SPOOF_STATIC_Z_STD_THRESHOLD, SPOOF_FOURIER_PEAK_THRESHOLD
)

LEFT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
RIGHT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
NOSE_TIP_INDEX = 1

class FaceAnalyzer:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        self.ear_history = deque(maxlen=LIVENESS_HISTORY_LENGTH)
        self.head_move_history = deque(maxlen=LIVENESS_HISTORY_LENGTH)

    def calculate_ear(self, eye_landmarks):
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5]); v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4]); h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return (v1 + v2) / (2.0 * h) if h > 0 else 0.0

    def fourier_analysis(self, face_crop):
        if face_crop.size == 0: return 0.0
        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY); f = np.fft.fft2(gray); fshift = np.fft.fftshift(f); magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1)
        h, w = magnitude_spectrum.shape; center_x, center_y = h // 2, w // 2; cv2.circle(magnitude_spectrum, (center_y, center_x), 5, 0, -1)
        return np.max(magnitude_spectrum)

    def analyze_frame(self, frame):
        frame_height, frame_width, _ = frame.shape
        frame.flags.writeable = False
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        frame.flags.writeable = True

        if not results.multi_face_landmarks:
            self.ear_history.clear(); self.head_move_history.clear()
            return {"status": "NO_FACE", "reason": "No face detected"}, None

        landmarks_obj = results.multi_face_landmarks[0]
        landmarks_list = landmarks_obj.landmark
        all_landmarks = np.array([(lm.x * frame_width, lm.y * frame_height, lm.z) for lm in landmarks_list])
        
        result_data = {} 

        x_coords, y_coords = all_landmarks[:, 0], all_landmarks[:, 1]
        xmin, xmax = int(np.min(x_coords)), int(np.max(x_coords))
        ymin, ymax = int(np.min(y_coords)), int(np.max(y_coords))
        padding = 10
        face_crop = frame[max(0, ymin-padding):min(ymax+padding, frame_height), max(0, xmin-padding):min(xmax+padding, frame_width)]
        
        result_data['fourier_peak'] = self.fourier_analysis(face_crop)
        result_data['z_std'] = np.std(all_landmarks[:, 2])
        
        print(f"DEBUG: Fourier Peak: {result_data['fourier_peak']:.2f} (threshold: {SPOOF_FOURIER_PEAK_THRESHOLD})")
        print(f"DEBUG: Z-Std: {result_data['z_std']:.4f} (threshold: {SPOOF_STATIC_Z_STD_THRESHOLD})")
        
        if result_data['fourier_peak'] > SPOOF_FOURIER_PEAK_THRESHOLD:
            print("DEBUG: BLOCKED - Fourier analysis detected screen/video")
            result_data.update({"status": "SPOOF", "reason": "Image/Video Not Allowed"})
            return result_data, landmarks_list

        if result_data['z_std'] < SPOOF_STATIC_Z_STD_THRESHOLD:
            print("DEBUG: BLOCKED - Z-depth too flat (static image)")
            result_data.update({"status": "SPOOF", "reason": "Static Image Not Allowed"})
            return result_data, landmarks_list

        left_ear = self.calculate_ear(all_landmarks[LEFT_EYE_INDICES, :2]); right_ear = self.calculate_ear(all_landmarks[RIGHT_EYE_INDICES, :2])
        self.ear_history.append((left_ear + right_ear) / 2.0)
        self.head_move_history.append(all_landmarks[NOSE_TIP_INDEX][2])

        if len(self.ear_history) < LIVENESS_HISTORY_LENGTH:
            result_data.update({"status": "ANALYZING", "reason": "Collecting Liveness Data", "ear_std": 0, "head_move_std": 0})
            return result_data, landmarks_list
        
        result_data['ear_std'] = np.std(self.ear_history); result_data['head_move_std'] = np.std(self.head_move_history)
        is_blinking = result_data['ear_std'] > LIVENESS_EAR_STD_THRESHOLD
        is_moving = result_data['head_move_std'] > LIVENESS_HEAD_MOVE_STD_THRESHOLD
        
        print(f"DEBUG: EAR Std: {result_data['ear_std']:.4f} (threshold: {LIVENESS_EAR_STD_THRESHOLD}) - Blinking: {is_blinking}")
        print(f"DEBUG: Head Move Std: {result_data['head_move_std']:.4f} (threshold: {LIVENESS_HEAD_MOVE_STD_THRESHOLD}) - Moving: {is_moving}")

        if is_blinking and is_moving:
            print("DEBUG: SUCCESS - Live person authenticated")
            result_data.update({"status": "REAL", "reason": "Live Person Authenticated"})
            return result_data, landmarks_list
        
        reason = ""
        if not is_blinking: reason += "Blink fail. ";
        if not is_moving: reason += "Movement fail.";
        print(f"DEBUG: BLOCKED - Liveness check failed: {reason.strip()}")
        result_data.update({"status": "SPOOF", "reason": reason.strip()})
        return result_data, landmarks_list

    def close(self):
        self.face_mesh.close()