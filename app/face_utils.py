import face_recognition
import numpy as np
from config import FACE_MATCH_THRESHOLD

def compute_encodings_for_locations(frame, known_locations):
    face_encodings = face_recognition.face_encodings(frame, known_locations)
    return face_encodings

def find_best_match(known_face_encodings, known_face_names, face_encoding_to_check):
    if not known_face_encodings:
        return "Unknown"
        
    matches = face_recognition.compare_faces(
        known_face_encodings,
        face_encoding_to_check,
        tolerance=FACE_MATCH_THRESHOLD
    )
    
    name = "Unknown"

    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding_to_check)
    if len(face_distances) > 0:
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]
            
    return name