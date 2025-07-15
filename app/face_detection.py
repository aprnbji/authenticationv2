import cv2

class FaceDetector:
    def __init__(self, model_path="models/haarcascade_frontalface_default.xml"):
        self.face_cascade = cv2.CascadeClassifier(model_path)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        return faces
