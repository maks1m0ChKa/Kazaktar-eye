from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware  # Импортируем CORS middleware
import torch
import uvicorn

app = FastAPI()

# Разрешаем CORS для всех источников (можешь настроить нужные домены в будущем)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000/"],
    allow_credentials=True,
    allow_methods=["POST"],
    allow_headers=["*"],
)

model = torch.hub.load('.', 'custom', path='yolov5s.pt', source='local')

@app.post("/home/")
async def home(file: UploadFile = File(...)):
    # Загружаем изображение
    img = await file.read()

    # Прогоняем через модель
    results = model(img)

    # Возвращаем результаты
    return results.pandas().xyxy[0].to_dict(orient="records")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
