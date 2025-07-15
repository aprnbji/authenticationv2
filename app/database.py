import redis
import json

class FaceDatabase:
    def __init__(self, host="localhost", port=6379, db=0):
        self.r = redis.Redis(host=host, port=port, db=db, decode_responses=True)

    def add_embedding(self, user_id, embedding):
        self.r.set(user_id, json.dumps(embedding))

    def get_embedding(self, user_id):
        data = self.r.get(user_id)
        return json.loads(data) if data else None

    def delete_embedding(self, user_id):
        self.r.delete(user_id)

    def list_users(self):
        return self.r.keys("*")
