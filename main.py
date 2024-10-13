from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import shutil
import os
from ultralytics import YOLO
from fastapi.responses import JSONResponse

app = FastAPI()

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загружаем модель напрямую с помощью пакета Ultralytics
model = YOLO('yolov8n.pt')

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Загружаем изображение
        img = await file.read()

        # Прогоняем через модель
        results = model(img)

        # Возвращаем результаты в формате JSON
        predictions = results.pandas().xyxy[0].to_dict(orient="records")
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        # Обработка ошибок
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.post("/train/")
async def train(dataset: UploadFile = File(...)):
    try:
        # Сохраните полученные данные
        dataset_path = "datasets/" + dataset.filename
        with open(dataset_path, "wb") as buffer:
            shutil.copyfileobj(dataset.file, buffer)


        os.system(f"yolo train model=yolov8n.pt data={dataset_path} epochs=50")

        return {"message": "Обучение завершено"}
    except Exception as e:
        # Обработка ошибок
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    if not os.path.exists("datasets/"):
        os.makedirs("datasets/")
    uvicorn.run(app, host="0.0.0.0", port=8000)
