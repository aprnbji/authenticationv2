import cv2
from app.database import FaceDatabase
from app.face_embeddings import DeepFaceEmbedder
from app.face_detection import FaceDetector

def main():

    detector = FaceDetector()
    embedder = DeepFaceEmbedder(model_name="Facenet512")
    db = FaceDatabase()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Could not open webcam.")
        return

    print("Press 's' to save a face and 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame.")
            break

        faces = detector.detect_faces(frame)
        
        if len(faces) == 1:
            (x, y, w, h) = faces[0]
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, "-.-", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        else:
            if len(faces) > 1:
                message = "Too many faces detected"
            else:
                message = "No face detected."
            cv2.putText(frame, message, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Face Registration", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            if len(faces) == 1:
                user_id = input("\nEnter user ID: ")

                if not user_id:
                    print("User ID cannot be empty.")
                    continue
                if db.get_embedding(user_id) is not None:
                    print(f"User ID '{user_id}' already exists. Please use a unique ID.")
                    continue

                (x, y, w, h) = faces[0]
                face_roi = frame[y:y+h, x:x+w]

                try:
                    embedding = embedder.get_embedding(face_roi)
                    
                    db.add_embedding(user_id, embedding)
                    
                    print(f"'{user_id}' has been added to the database.")

                except Exception as e:
                    print(f"Could not generate or save embedding.")
            else:
                print("could not save embedding.")

        elif key == ord('q'):
            print("Quitting application.")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()