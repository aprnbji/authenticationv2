from deepface import DeepFace
import cv2
import numpy as np

class DeepFaceEmbedder:
    def __init__(self, model_name="Facenet512"):
        self.model_name = model_name
        self.model = DeepFace.build_model(model_name)

    def get_embedding(self, face_img):
        embedding_obj = DeepFace.represent(
            img_path=face_img,
            model_name=self.model_name,
            enforce_detection=False  
        )
        
        embedding = embedding_obj[0]["embedding"]
        return embedding