from typing import Tuple, List
from PIL import Image
from io import BytesIO
from urllib.request import urlopen

import numpy as np
from base64 import b64decode
import cv2
from pydantic import BaseModel
from onnxruntime import InferenceSession


class Data(BaseModel):
    """
    Класс для валидации картинки из фронтенда.
    """
    image: str


def data_uri_to_cv2_img(uri: str) -> np.ndarray:
    """
    Преобразует URI изображения, закодированного в формате data URI, 
    в объект изображения OpenCV.

    Parameters
    ----------
    uri : str
        URI изображения в формате data URI, содержащий закодированные данные изображения.

    Returns
    -------
    img : numpy.ndarray 
        Объект изображения OpenCV.
    """

    with urlopen(uri) as response:
        _image = response.read()

    _bitmap = BytesIO(_image)
    img = Image.open(_bitmap)
    # добавление белого фона
    new_img = Image.new("RGBA", img.size, (255, 255, 255, 255))
    new_img.paste(img, (0, 0), img)
    img = new_img.convert("RGB")
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    return img


def detection_digits(img: np.ndarray) -> List:
    """
    Функция определяет положение цифр на картинке и 
    возвращает их положение и размер в виде списка кортежей.

    Parameters
    ----------
    img: np.ndarray
        Картинка на которой нужно распознать цифры

    Returns
    -------
    List - список кортежей отсортированный по x, [(x, y, w, h), ...], где 
        x - координата на оси x;
        y - координата на оси y;
        w - ширина;
        h - высота.
    """

    min_digit_size = 10
    max_digit_size = 400

    # преобразования и поиск контуров
    _, thresh = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if min_digit_size < w < max_digit_size and min_digit_size < h < max_digit_size:
            digit = x, y, w, h
            result.append(digit)
    # сортировка по X
    result = list(sorted(result, key=lambda digit_shape: digit_shape[0]))

    return result


def preprocessing(img: np.ndarray) -> np.ndarray:
    """
    Функция обработки картинки перед подачей в модель.

    Parameters
    ----------
    img: np.ndarray
        Картинка которую нужно обработать.

    Returns
    -------
    np.ndarray - обработанная картинка.
    """

    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV)
    norm_img = cv2.normalize(thresholded.astype(np.float32), None, 0.0, 1.0, cv2.NORM_MINMAX)
    digit = cv2.resize(norm_img, (28, 28), interpolation=cv2.INTER_AREA)

    return digit.reshape(1, 1, 28, 28)



def predict(model: InferenceSession, input_img: np.ndarray) -> int:
    """
    Функция делает прогноз с помощью загруженной модели ONNX. 
    Возвращает предсказанное число.

    Parameters
    ----------
    model: onnxruntime.InferenceSession
        Модель загруженная через ONNX.
    input_img: np.ndarray
        Картинка с цифрой, которую нужно распознать.

    Returns
    -------
    int - предсказанная цифра.
    """

    input_name = model.get_inputs()[0].name
    output_name = model.get_outputs()[0].name
    output = model.run([output_name], {input_name: input_img})
    y_pred = np.argmax(output[0])

    return y_pred
