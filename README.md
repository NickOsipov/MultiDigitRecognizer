# Multi-Digit Recognizer

В данном репозитории подготовлен проект для распознавания рукописных чисел.  
Распознавание производится с помощью сверточной нейронной сети (PyTorch) обученной на датасете MNIST и инструментов библиотеки OpenCV. Квантизация и сохранение готовой модели произведена с помощью ONNX. Подробное описание обучения модели и архитектура представлены в ноутбуке `notebooks/training.ipynb`.

## Используемые технологии
|         |            |        |
| -       | -          | -      |
| Python  | JavaScript | HTML   |
| PyTorch | NumPy      | OpenCV |
| ONNX    | FastAPI    | Docker |

## Структрура проекта

```
├── config                <- Директория с файлами конфигурации.
|   └── config.yaml       <- Файл конфигурации в формате .yaml
|
├── models                <- Директория с артефактами моделей.
│   ├── model.pt          <- Веса обученной модели, исходная версия fp32.
|   ├── model.py          <- Архитектура модели на PyTorch.
|   └── model.quant.onnx  <- Квантизированная модель в формате ONNX.
|
├── notebooks             <- Директория с Jupyter notebooks. 
│   └── training.ipynb    <- Ноутбук с обучением модели.
│
├── static                <- Директория с файлами фронтэнда.
|   ├── index.html        <- Шаблон HTML страницы.
|   └── script.js         <- JS код для кнопок и рисования чисел.
|
├── .dockerignore         <- Список игнорируемых файлов и директорий при сборке образа.
|
├── .gitignore            <- Список игнорируемых файлов и директорий при обновлении репозитория.
|
├── Dockerfile            <- Файл для сборки docker-image.
|
├── README.md             <- Описание репозитория.
|
├── __init__.py           <- Инициализация пакета.
|
├── app.py                <- Скрипт с бэкэндом на FastAPI.
|
├── requirements.txt      <- Файл с зафисимостями. 
|
├── run.sh                <- Скрипт для сборки docker-image и запуска котейнера.
│
└── utils.py              <- Скрипт с дополнительными функциями по обработке изображений.
```


## Описание проекта
Для рисования чисел подготовлен небольшой фронтэнд на HTML и JS. После запуска контейнера, в браузере можно открыть страницу, в которой будет поле для рисования, кнопки **Распознать** и **Очистить**, а также строка с результатом распознавания. 

Распознавание производится по следующему сценарию:
1. После нажатия кнопки **Распознать** картинка из поля на фронтенде, отправляется на основной эндпоинт `recognize`.
2. Происходит декодирование картинки из `base64` в формат для OpenCV.
3. Детектируется каждая цифра по отдельности и сохраняется список с координатами и размерами.
4. Для каждой цифры проводится небольшая предобработка для подготовки к использованию в модели.
5. Модель классифицирует цифру.
6. Все цифры собираются в число которое отправляется обратно на фронтэнд в строку Результат.

## Инструкция по запуску
Инструкция предсталена для ОС Linux. Проект собран с помощью docker, поэтому предварительно необходимо [установить](https://docs.docker.com/engine/install/ubuntu/). 
1. Запускаем `run.sh` в терминале. Соберется docker-image `mdrecognizer` и поднимется в интерактивном режиме docker-contatiner `mdr_container`. После остановки контейнер удаляется.  
```bash
bash run.sh
```
2. Скопируем в браузер следующий адрес, чтобы открыть поле для рисования.
```html
http://127.0.0.1/static/index.html
```
3. Рисуем числа и нажимаем кнопку **Распознать**. Чтобы стереть изображение нажимаем **Очистить**.

## Примечание
Для уменьшения размера docker образа принято решение не включать в сборку библиотеки torch и torchvision, так как модель запускается через ONNX, а обработка изображений происходит с помощью OpenCV и NumPy. Поэтому ноутбук с обучением можно просмотреть в Google Colab. Файлы `models/model.pt` и `models/model.py` представлены как артефакты и не используются при работе приложения.