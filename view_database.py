import redis
import json
import numpy as np
from config import REDIS_HOST, REDIS_PORT 

def view_database():
    print(f"Connecting to Redis at {REDIS_HOST}:{REDIS_PORT}...")
    
    try:
        r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, db=0, decode_responses=True)
        r.ping()
        print("Connection successful.\n")
    except redis.exceptions.ConnectionError as e:
        print(f"FATAL: Could not connect to Redis. Is it running? Details: {e}")
        return

    keys = list(r.scan_iter("user:*"))

    if not keys:
        print("No user data found in the database.")
    else:
        print(f"Found {len(keys)} user(s) in the database:")
        print("="*50)
        for key in keys:
            value = r.get(key)
            try:
                embedding_list = json.loads(value)
                embedding_array = np.array(embedding_list)
                
                print(f"👤 KEY: {key}")
                print(f"   TYPE: Face Embedding")
                print(f"   DIMENSIONS: {embedding_array.shape[0]} values")
                print(f"   SAMPLE: {embedding_array[:5]} ...") 
                
            except (json.JSONDecodeError, TypeError):
                print(f"🔑 KEY: {key}")
                print(f"   TYPE: Raw String")
                print(f"   VALUE: {value}")

            print("-" * 50)

if __name__ == "__main__":
    view_database()