import os
import sys
import time
import shutil
import uuid
import numpy as np
import cv2
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# —— Paths ——
if getattr(sys, 'frozen', False):
    RUNTIME_DIR = Path(sys._MEIPASS)
    EXE_DIR = Path(sys.executable).parent
    OUTPUT_DIR  = EXE_DIR / "output"
    MODELS_DIR  = EXE_DIR / "models"
    FRONTEND_DIR = RUNTIME_DIR / "frontend" / "dist"
else:
    RUNTIME_DIR = Path(__file__).parent.parent
    OUTPUT_DIR  = RUNTIME_DIR / "output"
    MODELS_DIR  = RUNTIME_DIR / "models"
    FRONTEND_DIR = RUNTIME_DIR / "frontend" / "dist"

OUTPUT_DIR.mkdir(exist_ok=True)
TEMP_BASE = Path(os.environ.get("TEMP", "/tmp"))

app = FastAPI(title="Optico API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost", "http://127.0.0.1", "http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

def safe_join(base: Path, path: str) -> Path:
    full_path = (base / path).resolve()
    if not str(full_path).startswith(str(base.resolve())):
        raise HTTPException(status_code=403, detail="Path traversal forbidden.")
    return full_path

jobs = {}

class ProcessConfig(BaseModel):
    stack_mode:    str   = "median"    
    super_res:     str   = "none"      
    mfsrScale:     float = 1.5         
    face_restore:  bool  = False
    sharpen:       float = 0.35
    brightness:    int   = 0
    contrast:      int   = 0
    ref_index:     int   = 0
    align_method:  str   = "ecc"       
    max_offset:    float = 20.0

from fastapi.staticfiles import StaticFiles

@app.get("/api/health")
async def health():
    return {"status": "ok", "version": "1.0.0"}

app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")

@app.post("/api/upload")
async def upload_images(files: list[UploadFile] = File(...)):
    job_id = uuid.uuid4().hex[:8]
    tmp_dir = TEMP_BASE / f"optico_{job_id}"
    tmp_dir.mkdir(exist_ok=True)
    
    filepaths = []
    for f in files:
        path = tmp_dir / f.filename
        with open(path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)
        filepaths.append(str(path))
        
    jobs[job_id] = {"tmp_dir": str(tmp_dir), "files": filepaths, "status": "uploaded"}
    return {"job_id": job_id, "files": [Path(p).name for p in filepaths]}

@app.get("/api/image/{job_id}/{filename}")
async def get_image(job_id: str, filename: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    tmp_dir = Path(jobs[job_id]["tmp_dir"])
    file_path = safe_join(tmp_dir, filename)
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    return FileResponse(str(file_path))

# —— Entry Point ——
if __name__ == "__main__":
    import uvicorn
    import threading
    import webview
    
    port = int(os.environ.get("OPTICO_PORT", "18765"))
    
    def start_server():
        uvicorn.run(
            "main:app",
            host="127.0.0.1",
            port=port,
            log_level="warning",
            access_log=False,
        )
        
    t = threading.Thread(target=start_server)
    t.daemon = True
    t.start()
    
    time.sleep(1.5)
    
    window = webview.create_window(
        title="Optico",
        url=f"http://127.0.0.1:{port}",
        width=1280,
        height=800,
        min_size=(1024, 768),
        background_color="#080A0F"
    )
    
    webview.start(debug=False)
