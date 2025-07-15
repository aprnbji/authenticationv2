import redis
import json

r = redis.Redis()

keys = r.keys("*")

if not keys:
    print("No data found in Redis.")
else:
    for key in keys:
        key_str = key.decode()  
        value = r.get(key)
        try:
            value_str = json.loads(value)
        except:
            value_str = value.decode()
        print(f"KEY: {key_str}\nVALUE: {value_str}\n{'-'*50}")
