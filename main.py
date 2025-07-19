import uvicorn
import cv2
import numpy as np
import base64
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from contextlib import asynccontextmanager
from pydantic import BaseModel

from app.liveness import FaceAnalyzer
from app.face_utils import compute_encodings_for_locations, find_best_match
from app.database import db

app_state = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Initializing system components at startup...")
    app_state["face_analyzer"] = FaceAnalyzer()
    print("FaceAnalyzer loaded.")
    yield
    print("Shutting down and cleaning up resources...")
    app_state["face_analyzer"].close()

class FrameData(BaseModel):
    mode: str
    image: str
    name: str | None = None

class UserDeleteData(BaseModel):
    name: str

app = FastAPI(lifespan=lifespan)
app.mount("/static", StaticFiles(directory="static"), name="static")

def base64_to_frame(base64_str):
    if "base64," in base64_str:
        base64_str = base64_str.split("base64,")[1]
    img_data = base64.b64decode(base64_str)
    np_arr = np.frombuffer(img_data, np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

@app.post("/api/process-frame")
async def process_frame_endpoint(data: FrameData):
    """
    Receives a frame via HTTP POST, processes it, and returns a rich result.
    """
    known_face_names, known_face_encodings = db.get_all_users()
    frame = base64_to_frame(data.image)
    face_analyzer = app_state["face_analyzer"]
    
    analysis_result, landmarks = face_analyzer.analyze_frame(frame)
    status = analysis_result.get("status")

    response = analysis_result.copy()
    response.update({"name": "UNVERIFIED", "location": None})
    
    face_location = None
    if landmarks:
        h, w, _ = frame.shape
        x_coords = [lm.x * w for lm in landmarks]; y_coords = [lm.y * h for lm in landmarks]
        face_location = (int(min(y_coords)), int(max(x_coords)), int(max(y_coords)), int(min(x_coords)))
        response["location"] = face_location

    if status == "REAL" and face_location:
        face_encodings = compute_encodings_for_locations(frame, [face_location])
        if face_encodings:
            face_encoding = face_encodings[0]
            if data.mode == "enroll":
                if data.name:
                    db.add_user(data.name, face_encoding)
                    response.update({"status": "ENROLLED", "name": data.name})
                else:
                    response.update({"status": "ERROR", "reason": "Name is required for enrollment."})
            elif data.mode == "verify":
                matched_name = find_best_match(known_face_encodings, known_face_names, face_encoding)
                response.update({"status": "VERIFIED", "name": matched_name})
        else:
            response.update({"status": "ERROR", "reason": "Could not compute face encoding."})

    return response

@app.get("/api/users")
async def get_users():
    """Returns a list of all enrolled users."""
    names, _ = db.get_all_users()
    return {"users": names}

@app.post("/api/delete-user")
async def delete_user_endpoint(data: UserDeleteData):
    """Deletes a specific user from the database."""
    db.delete_user(data.name)
    return {"message": f"User '{data.name}' deleted successfully."}

@app.post("/api/clear-database")
async def clear_database():
    """Clears all users from the database."""
    db.clear_all()
    return {"message": "Database cleared successfully."}

@app.get("/", response_class=HTMLResponse)
async def read_root():
    with open("static/index.html") as f:
        return f.read()

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)