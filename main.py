from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from yolov5.inference import detect
import torch
import uvicorn
import sys
sys.path.append('/path/to/yolov5')
from yolov5.detect import run as yolo_detect


app = FastAPI()

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5:master', 'yolov5s', pretrained=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Save the uploaded file locally
        with open("image.jpg", "wb") as buffer:
            buffer.write(file.file.read())

        # Perform inference
        results = detect.run(weights='best.pt', imgsz=640, source='image.jpg', device='cpu')

        # Extract predictions
        predictions = results.xyxy[0].tolist()

        return JSONResponse(content=predictions, status_code=200)

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
