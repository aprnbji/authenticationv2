import redis
import numpy as np
import json
from config import REDIS_HOST, REDIS_PORT

class FaceDatabase:
    def __init__(self, host=REDIS_HOST, port=REDIS_PORT, db=0):
        try:
            self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)
            self.r.ping()
            print("Successfully connected to Redis.")
        except redis.exceptions.ConnectionError as e:
            print(f"FATAL: Could not connect to Redis at {host}:{port}. Is it running?")
            print(f"Error details: {e}")
            exit()

    def add_user(self, name, embedding):
        key = f"user:{name}"
        value = json.dumps(embedding.tolist())
        self.r.set(key, value)
        print(f"User '{name}' added to the database.")

    def get_all_users(self):
        known_face_names = []
        known_face_encodings = []
        
        for key in self.r.scan_iter("user:*"):
            name = key.split(":")[1]
            data = self.r.get(key)
            
            if data:
                embedding_list = json.loads(data)
                embedding = np.array(embedding_list)
                
                known_face_names.append(name)
                known_face_encodings.append(embedding)

        return known_face_names, known_face_encodings

    def delete_user(self, name):
        key = f"user:{name}"
        if self.r.delete(key):
            print(f"User '{name}' deleted from the database.")
        else:
            print(f"User '{name}' not found in the database.")

db = FaceDatabase()