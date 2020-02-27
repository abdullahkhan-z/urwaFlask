
from PIL import Image
import requests
import os
from io import BytesIO
from app.config import *

def get_PIL_image_path(image_path):
     return Image.open(image_path)

def get_PIL_image_url(image_url):
    response = requests.get(image_url)
    img = Image.open(BytesIO(response.content))
    return img


def download_model(path_to_model, model_url):
    if not os.path.exists('models'):
        os.makedirs('models', exist_ok=True)
    if not os.path.exists(path_to_model):
        print('model weights were not found, downloading them...')
        r = requests.get(model_url)
        open(path_to_model, 'wb').write(r.content)
    else:
        print('model weights found.')
        

def download_artifacts():
    print("Image Gender Model:")
    download_model(IMAGE_GENDER_MODEL_PATH, IMAGE_GENDER_MODEL_URL)
    print("Image Age Model:")
    download_model(IMAGE_AGE_MODEL_PATH, IMAGE_AGE_MODEL_URL)
    print("Image Race Model:")
    download_model(IMAGE_RACE_MODEL_PATH, IMAGE_RACE_MODEL_URL)
    print("Name Gender Model:")
    download_model(NAME_GENDER_MODEL_PATH, NAME_GENDER_MODEL_URL)
    print("Name Gender Lookup:")
    download_model(NAME_GENDER_DICT_PATH, NAME_GENDER_DICT_URL)