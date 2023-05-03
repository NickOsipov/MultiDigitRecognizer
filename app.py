import cv2
import numpy as np
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import yaml
from onnxruntime import InferenceSession

from utils import (
    Data, data_uri_to_cv2_img, detection_digits, preprocessing, predict
)


app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")


# загрузка конфига
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# загрузка модели
model_onnx_path = config.get("model_onnx_path", "")
model = InferenceSession(model_onnx_path)


@app.get("/")
def check_status() -> dict:
    """Проверка старта сервиса."""
    return {"status": "ok"}


@app.post("/recognize/")
def main(data: Data) -> dict:
    """
    Основной endpoint для распознавания. 
    Возвращает словарь со значением распознанного числа. 
    
    Parameters
    ----------
    data : Data
        Картинка в формате base64.

    Returns
    -------
    dict - словарь со значением распознанного числа.
    """
    # достаем картинку из base64 и находим координаты и размеры
    img = data_uri_to_cv2_img(data.image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    digits = detection_digits(img)

    # если список digits не пустой то проходим по нему в цикле
    # образаем цифру по одной
    # обрабатываем и распознаем
    # собираем цифры в число и сохраняем формате строки
    if digits:
        result = ""
        for digit in digits:
            x, y, w, h = digit
            tmp = img[y:y+h, x:x+w]
            tmp = preprocessing(tmp)
            y_pred = predict(model, tmp)
            result += str(y_pred)

        return {"value": result}
    else:
        return {"value": "Пустая картинка"}
