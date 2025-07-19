import cv2
import numpy as np
from config import DEPTH_SPOOF_STD_DEV_THRESHOLD

class DepthEstimator:
    def __init__(self):
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.frame_count = 0

    def texture_analysis(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        grad_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        texture_std = np.std(gradient_magnitude)
        texture_mean = np.mean(gradient_magnitude)
        
        return texture_std, texture_mean

    def frequency_analysis(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        
        magnitude_spectrum = np.log(np.abs(f_shift) + 1)
        
        freq_std = np.std(magnitude_spectrum)
        
        return freq_std

    def edge_analysis(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        edges = cv2.Canny(gray, 50, 150)
        
        edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
        
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        contour_complexity = 0
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            if cv2.contourArea(largest_contour) > 1000:  
                perimeter = cv2.arcLength(largest_contour, True)
                area = cv2.contourArea(largest_contour)
                if area > 0:
                    contour_complexity = perimeter / area
        
        return edge_density, contour_complexity

    def motion_consistency(self, frame):
        fg_mask = self.bg_subtractor.apply(frame)
        
        motion_pixels = np.sum(fg_mask > 0)
        total_pixels = fg_mask.shape[0] * fg_mask.shape[1]
        motion_ratio = motion_pixels / total_pixels
        
        return motion_ratio

    def check_spoofing(self, frame):
        self.frame_count += 1
        
        if self.frame_count < 5:
            return "ANALYZING"
        
        try:
            texture_std, texture_mean = self.texture_analysis(frame)
            
            freq_std = self.frequency_analysis(frame)
            
            edge_density, contour_complexity = self.edge_analysis(frame)
            
            motion_ratio = self.motion_consistency(frame)
            
            real_indicators = 0
            total_indicators = 0
            
            if texture_std > 15.0:  
                real_indicators += 1
            total_indicators += 1
            
            if freq_std > 8.0:  
                real_indicators += 1
            total_indicators += 1
            
            if edge_density > 0.05 and contour_complexity > 0.1:  
                real_indicators += 1
            total_indicators += 1
            
            if self.frame_count > 10:
                if 0.001 < motion_ratio < 0.1:  
                    real_indicators += 1
                total_indicators += 1
            
            confidence = real_indicators / total_indicators
            
            if confidence >= 0.6:  
                return "REAL"
            else:
                return "SPOOF"
                
        except Exception as e:
            print(f"Error in depth estimation: {e}")
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            simple_std = np.std(gray)
            return "REAL" if simple_std > 20 else "SPOOF"

    def reset(self):
        """Reset the background subtractor and frame count."""
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(detectShadows=True)
        self.frame_count = 0